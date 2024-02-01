import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math
import argparse
from .common import *
from .libs.pls.parallel_local_search import PLSExplainer


def np_woe(p):
    return np.log(p / (1-p))

# define objective function here. will use this to define an f(x) objective function to give to PLSExplainer class
def objective_function(args, explanation, always_keep_idx, task_model, orig_pred_prob, orig_input_kwargs):
    '''
    computes suff or comp objective on the current explanation                
    - explanation : np.ndarray of shape (r,d) where r is number of parallel runs and d is number of tokens (excluding always_keep tokens described next). 
    - always_keep_idx : binary array of len d', where d' is the number of tokens including tokens to be always kept, like special tokens
    - task_model: classifier that returns tuple with logits as SECOND element (after e.g. the loss)
    - orig_pred_prob: the predicted probability (for the predicted class) of the classifier on the current data point
    - orig_input_kwargs: dict with kwargs used to obtain orig_pred_prob, including 'attention_mask' item
    returns the suff/comp of task_model computed on document using attention_mask=mask, as well as the weight of evidence version of the objective
    ''' 
    # split input into batches if too many parallel runs (i.e. parallel states = parallel data points) to fit into memory
    # import pdb; pdb.set_trace()
    # batch_size = min(args.batch_size, len(explanation))
    num_explanations = len(explanation)
    num_batches = max(1, math.ceil(min(args.num_restarts, num_explanations) / args.batch_size)) # num_restarts is num parallel runs
    num_tokens = orig_input_kwargs['input_ids'].size(-1)
    explanations = np.array_split(explanation, indices_or_sections=num_batches)
    obj_vals, obj_val_woes = [], []
    # repeat original data point batch_size times
    stacked_kwargs = {
        k : v.expand([num_explanations] + list(v.shape)[-1:]).squeeze(-1).clone() for k,v in orig_input_kwargs.items() # clone to reallocate memory / squeeze() for labels special case
    }
    # get eligible for removal idx
    eligible_for_removal = torch.ones(num_tokens).to(args.device)
    eligible_for_removal[always_keep_idx] = 0.
    eligible_for_removal_idx = torch.where(eligible_for_removal)[0]
    orig_pred_prob = orig_pred_prob.detach().cpu().numpy()
    for explanation in explanations:
        new_attention_mask = torch.tensor(explanation).long().to(args.device)
        stacked_kwargs['attention_mask'][:,eligible_for_removal_idx] = new_attention_mask
        outputs = task_model(**stacked_kwargs)
        pred_probs = torch.softmax(outputs[1], dim=-1)
        pred_prob = torch.gather(pred_probs, 1, stacked_kwargs['labels'].reshape(-1,1)).detach().cpu().numpy()
        if args.objective == 'suff':
            obj_val = orig_pred_prob - pred_prob
            obj_val_woe = np_woe(orig_pred_prob) - np_woe(pred_prob)
        if args.objective == 'comp':
            obj_val = -(orig_pred_prob - pred_prob)
            obj_val_woe = -(np_woe(orig_pred_prob) - np_woe(pred_prob))
        obj_vals.append(obj_val)
        obj_val_woes.append(obj_val_woe)
    obj_val = np.concatenate(obj_vals).reshape(-1)
    obj_val_woe = np.concatenate(obj_val_woes).reshape(-1)
    return obj_val, obj_val_woe


class PLSTextCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model, tokenizer, gpu=0, seed=0, explanation_sparsity=0.2, num_search=1000, batch_size=10, 
                 num_restarts=10, objective="suff", special_tokens=None):
        class Args:
            def __init__(self, gpu=0, seed=0, explanation_sparsity=0.2, num_search=1000, batch_size=10, 
                 num_restarts=10, objective="suff"):
                super().__init__()
                self.gpu = gpu
                self.seed = seed
                self.explanation_sparsity = explanation_sparsity
                self.num_search = num_search
                self.batch_size = batch_size
                self.num_restarts = num_restarts
                self.objective = objective
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Args with parameters
        self.pls_args = Args(gpu=gpu, seed=seed, explanation_sparsity=explanation_sparsity, num_search=num_search, 
                         batch_size=batch_size, num_restarts=num_restarts, objective=objective)
        print(self.pls_args)

        self.pls_args=Args()
        if special_tokens is None:
            self.special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
        else:
            self.special_tokens = special_tokens

        # model_wrapper = BertWrapperTorch(model, device)
        super().__init__(model)
        self.tokenizer = tokenizer

    def forward(self, x, t, **kwargs):
        device = x.device
        bsz = x.shape[0]

        if not isinstance(t, torch.Tensor) and t is not None:
            t = torch.tensor(t)

        masks_all = []
        mask_weights_all = []
        explanation_strs = []
        for i in range(bsz):
            #{k: v for k, v in kwargs.items()}
            model_kwargs = {k: v[i].unsqueeze(0) for k, v in kwargs.items()}
            model_kwargs['input_ids'] = x[i].unsqueeze(0)
            model_kwargs['labels'] = t[i].unsqueeze(0)

            always_keep_idx = np.argwhere([_id in self.special_tokens for _id in model_kwargs['input_ids'].squeeze()]).reshape(-1)
            always_keep_idx = torch.Tensor(always_keep_idx).long().to(device)
            num_spcl_tokens = len(always_keep_idx)
            search_dimensionality = model_kwargs['input_ids'].shape[-1] - num_spcl_tokens

            def current_objective_function(explanation):
                return objective_function(
                    args=self.pls_args, 
                    explanation=explanation, 
                    always_keep_idx=always_keep_idx,
                    task_model=self.model, 
                    orig_pred_prob=t[i], 
                    orig_input_kwargs=model_kwargs
                )


            PLS = PLSExplainer(
                    objective_function=current_objective_function, 
                    target_sparsity=self.pls_args.explanation_sparsity, 
                    eval_budget=self.pls_args.num_search,
                    dimensionality=search_dimensionality,
                    restarts=self.pls_args.num_restarts, # num parallel runs
                    temp_decay=0, # temp in simulated annealing (0 <= x <= 1). set to higher value for more exploration early in search
                    search_space='exact_k',
                    no_duplicates=True # avoid duplicate evaluations of objective function
                )
            
            explanations, obj_values, obj_woe_values = PLS.run()
            best_explanation = explanations.tolist()[-1]
            masks_all.append(best_explanation)
            mask_weights_all.append(obj_values[-1])

            input_ids = model_kwargs['input_ids'].squeeze().cpu().tolist()
            model_input_str = self.tokenizer.decode([_id for _id in input_ids if _id != self.tokenizer.pad_token_id])
            eligible_ids = [_id for i, _id in enumerate(input_ids) if i not in always_keep_idx]
            blank_id = self.tokenizer.encode(' __ ', add_special_tokens=False)[0]
            explanation_tokens = [_id if best_explanation[i] == 1 else blank_id for i, _id in enumerate(eligible_ids)]
            explanation_str = self.tokenizer.decode(explanation_tokens)
            explanation_strs.append(explanation_str)
        
        # pad explanations to max length
        masks_all = [m + [0]*(x.shape[-1] - len(m)) for m in masks_all]
        masks_all = torch.tensor(masks_all).to(device)
        mask_weights_all = torch.tensor(mask_weights_all).to(device)

        return FeatureAttrOutput(masks_all * mask_weights_all, {
            "expln_flat_masks": masks_all,
            "masks": masks_all,
            "mask_weights": mask_weights_all,
            "explanation_strs": explanation_strs
        })