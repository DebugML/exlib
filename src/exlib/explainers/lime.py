import torch
import torch.nn.functional as F
import lime
import math
import numpy as np
from lime import lime_image, lime_text, lime_base
from .common import *
from .libs.lime.lime_text import LimeTextExplainer


def lime_cls_closure(model, collapse):
    def go(x_np):
        x = np_to_torch_img(x_np)
        x = x.to(next(model.parameters()).device)
        if collapse:
            x = x[:,0:1,:,:] # even though lime needs no singleton last dimension in its input,
            # for an odd reason they put back 3 of them to match RGB format before passing
            # to batch_predict. So we need to remove the extraneous ones.
        y = model(x)
        return y.detach().cpu().numpy()
    return go


def explain_image_cls_with_lime(model, x, ts,
                                LimeImageExplainerKwargs={},
                                # Gets FA for every label if top_labels == None
                                explain_instance_kwargs={},
                                get_image_and_mask_kwargs={},
                                return_groups=False
                                ):
    """
    Explain a pytorch model with LIME.
    this function is not intended to be called directly.
    We only explain one image at a time.

    # LimeImageExplainer args
    kernel_width=0.25, kernel=None, verbose=False, feature_selection='auto', random_state=None

    # explain_instance args
    image, classifier_fn, labels=(1, ), hide_color=None, top_labels=5,
    num_features=100000, num_samples=1000, batch_size=10, segmentation_fn=None,
    distance_metric='cosine', model_regressor=None, random_seed=None, progress_bar=true

    # get_image_and_mask arguments
    positive_only=true, negative_only=False, hide_rest=False, num_features=5, min_weight=0.0
    """

    ## Images here are not batched
    C, H, W = x.shape
    x_np = x.cpu().permute(1,2,0).numpy()

    collapse = x.size(0) == 1
    if collapse:
        x_np = x_np[:,:,0]

    f = lime_cls_closure(model, collapse)
    explainer = lime_image.LimeImageExplainer(**LimeImageExplainerKwargs)

    if isinstance(ts, torch.Tensor):
        todo_labels = ts.numpy()
    else:
        todo_labels = ts

    lime_exp = explainer.explain_instance(x_np, f, labels=todo_labels, **explain_instance_kwargs)

    # Initialize tensors for attributions and masks
    seg_mask = torch.from_numpy(lime_exp.segments).to(x.device)
    attrs_all = torch.zeros((len(todo_labels), H, W), device=x.device)
    group_masks_all = torch.zeros((len(todo_labels), H, W), dtype=torch.long, device=x.device)
    group_attrs_all = torch.zeros((len(todo_labels), len(seg_mask.unique())), dtype=torch.float, 
        device=x.device)

    # Vectorized operation to handle multiple labels and segmentations
    seg_attrs_all = [torch.tensor(lime_exp.local_exp[ti], device=x.device) for ti in todo_labels]

    for i, seg_attrs in enumerate(seg_attrs_all):
        seg_ids = seg_attrs[:, 0].long()  # Segment IDs
        seg_values = seg_attrs[:, 1].float()  # Corresponding attribution values
        
        # import pdb; pdb.set_trace()
        # Vectorized addition of attributions based on segments
        attrs_all[i] = ((seg_mask.unsqueeze(0) == seg_ids.view(-1, 1, 1)).float() \
            * seg_values.view(-1, 1, 1)).sum(dim=0)
        
        if return_groups:
            # Assign group mask (vectorized)
            group_masks_all[i] = torch.where(seg_mask.unsqueeze(0) == seg_ids.view(-1, 1, 1), 
                    torch.arange(len(seg_ids)).to(x.device).view(-1, 1, 1), 
                    group_masks_all[i]).sum(dim=0).long()
            group_attrs_all[i] = seg_values

    attrs_all = attrs_all.permute(1,2,0)[None]  # (1, H, W, N)
    group_masks_all = group_masks_all.permute(1,2,0)[None]  # (1, H, W, N)
    group_attrs_all = group_attrs_all.permute(1,0) # (M, N)

    if return_groups:
        return GroupFeatureAttrOutput(attrs_all, lime_exp, group_masks_all, group_attrs_all)
    else:
        return FeatureAttrOutput(attrs_all, lime_exp)


"""
    Explainer output format:
    - attributions: (N, C, H, W) or (N, C, H, W, T) or (N, 1, H, W) or (N, 1, H, W, T)
    - group_masks: (N, M, C, H, W) or (N, M, C, H, W, T) or (N, M, 1, H, W) or (N, M, 1, H, W, T)
    - group_attributions: (N, M) or (N, M, T)
"""
class LimeImageCls(FeatureAttrMethod):
    def __init__(self, model,
                 LimeImageExplainerKwargs={},
                 explain_instance_kwargs={
                     # Make this however big you need to get every label
                     # this is because the original LIME API is stupid
                     "top_labels" : 1000000,
                     "num_samples" : 500,
                 },
                 get_image_and_mask_kwargs={}):
        super(LimeImageCls, self).__init__(model)
        self.LimeImageExplainerKwargs = LimeImageExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs
        self.get_image_and_mask_kwargs = get_image_and_mask_kwargs

    def forward(self, x, t, return_groups=False):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        N = x.size(0)
        assert x.ndim == 4 and t.ndim in [1, 2] and len(t) == N
        if t.ndim == 1:
            t = t.unsqueeze(1)

        attrs, lime_exps = [], []
        group_masks, group_attrs = [], []
        for i in range(N):
            xi, ti = x[i], t[i].cpu().tolist()
            out = explain_image_cls_with_lime(self.model, xi, ti,
                    LimeImageExplainerKwargs=self.LimeImageExplainerKwargs,
                    explain_instance_kwargs=self.explain_instance_kwargs,
                    get_image_and_mask_kwargs=self.get_image_and_mask_kwargs,
                    return_groups=return_groups)

            attrs.append(out.attributions)
            lime_exps.append(out.explainer_output)
            if return_groups:
                group_masks.append(out.group_masks)
                group_attrs.append(out.group_attributions)

        attrs = torch.stack(attrs, dim=0)
        if return_groups:
            group_masks = torch.stack(group_masks, dim=0)
            group_attrs = torch.stack(group_attrs, dim=0)
        if attrs.ndim == 5 and attrs.size(-1) == 1:
            attrs = attrs.squeeze(-1)
            if return_groups:
                group_masks = group_masks.squeeze(-1)
                group_attrs = group_attrs.squeeze(-1)

        if return_groups:
            return GroupFeatureAttrOutput(attrs, lime_exps, group_masks, group_attrs)
        else:
            return FeatureAttrOutput(attrs, lime_exps)


# Segmentation model
class LimeImageSeg(FeatureAttrMethod):
    def __init__(self, model,
                 LimeImageExplainerKwargs={},
                 explain_instance_kwargs={
                     # Make this however big you need to get every label
                     # this is because the original LIME API is stupid
                     "top_labels" : 1000000,
                     "num_samples" : 500,
                 },
                 get_image_and_mask_kwargs={}):
        super(LimeImageSeg, self).__init__(model)
        self.LimeImageExplainerKwargs = LimeImageExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs
        self.get_image_and_mask_kwargs = get_image_and_mask_kwargs

        self.cls_model = Seg2ClsWrapper(model)

    def forward(self, x, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        N = x.size(0)
        assert x.ndim == 4 and t.ndim == 1 and len(t) == N

        attrs, lime_exps = [], []
        for i in range(N):
            xi, ti = x[i], t[i].cpu().item()
            out = explain_image_cls_with_lime(self.cls_model, xi, [ti],
                    LimeImageExplainerKwargs=self.LimeImageExplainerKwargs,
                    explain_instance_kwargs=self.explain_instance_kwargs,
                    get_image_and_mask_kwargs=self.get_image_and_mask_kwargs)

            attrs.append(out.attributions)
            lime_exps.append(out.explainer_output)

        return FeatureAttrOutput(torch.stack(attrs), lime_exps)



def lime_cls_closure_text(model, tokenizer):
    def go(x_str):
        x_raw = [tokenizer.decode(tokenizer.convert_tokens_to_ids(x_str_i.split())) for x_str_i in x_str]
        inp = tokenizer(x_raw, 
                        padding='max_length', 
                        truncation=True, 
                        max_length=512)
        inp = {k: torch.tensor(v).to(next(model.parameters()).device) 
               for k, v in inp.items()}
        y = model(**inp)
        return y.detach().cpu().numpy()
    return go

def explain_text_cls_with_lime(model, tokenizer, x, ts,
                                LimeTextExplainerKwargs={},
                                # Gets FA for every label if top_labels == None
                                explain_instance_kwargs={},
                                return_groups=False):
    """
    Explain a pytorch model with LIME.
    this function is not intended to be called directly.
    We only explain one text at a time.

    # LimeImageExplainer args
    kernel_width=0.25, kernel=None, verbose=False, feature_selection='auto', random_state=None

    # explain_instance args
    image, classifier_fn, labels=(1, ), hide_color=None, top_labels=5,
    num_features=100000, num_samples=1000, batch_size=10, segmentation_fn=None,
    distance_metric='cosine', model_regressor=None, random_seed=None, progress_bar=true

    # get_image_and_mask arguments
    positive_only=true, negative_only=False, hide_rest=False, num_features=5, min_weight=0.0
    """

    ## Texts here are not batched
    L, = x.shape

    tokens = tokenizer.convert_ids_to_tokens(x)
    x_str = ' '.join(tokens)

    f = lime_cls_closure_text(model, tokenizer)
    explainer = LimeTextExplainer(**LimeTextExplainerKwargs)

    if isinstance(ts, torch.Tensor):
        todo_labels = ts.numpy()
    else:
        todo_labels = ts

    lime_exp = explainer.explain_instance(x_str, f, labels=todo_labels, **explain_instance_kwargs)
    # import pdb; pdb.set_trace()
    # Initialize tensors for attributions and masks
    # seg_mask = torch.from_numpy(lime_exp.segments).to(x.device)
    attrs_all = torch.zeros((len(todo_labels), L), device=x.device)
    group_masks_all = torch.zeros((len(todo_labels), L), dtype=torch.long, device=x.device)
    group_attrs_all = torch.zeros((len(todo_labels), L), dtype=torch.float, 
        device=x.device)

    # Vectorized operation to handle multiple labels and segmentations
    seg_attrs_all = [torch.tensor(lime_exp.local_exp[ti], device=x.device) for ti in todo_labels]
    # print('seg_attrs_all')
    # import pdb; pdb.set_trace()

    for i, seg_attrs in enumerate(seg_attrs_all):
        seg_ids = seg_attrs[:, 0].long()  # Segment IDs
        seg_values = seg_attrs[:, 1].float()  # Corresponding attribution values
        
        # import pdb; pdb.set_trace()
        # Vectorized addition of attributions based on segments
        # attrs_all[i] = ((seg_mask.unsqueeze(0) == seg_ids.view(-1, 1, 1)).float() \
        #     * seg_values.view(-1, 1, 1)).sum(dim=0)
        
        attrs_all[i, seg_ids] = seg_values
        
        # if return_groups:
        #     # Assign group mask (vectorized)
        #     group_masks_all[i] = torch.where(seg_mask.unsqueeze(0) == seg_ids.view(-1, 1, 1), 
        #             torch.arange(len(seg_ids)).to(x.device).view(-1, 1, 1), 
        #             group_masks_all[i]).sum(dim=0).long()
        #     group_attrs_all[i] = seg_values

    # print('attrs_all')
    # import pdb; pdb.set_trace()
    # attrs_all = attrs_all.permute(1,2,0)[None]  # (1, H, W, N)
    # group_masks_all = group_masks_all.permute(1,2,0)[None]  # (1, H, W, N)
    # group_attrs_all = group_attrs_all.permute(1,0) # (M, N)

    # print('group_attrs_all')
    # import pdb; pdb.set_trace()

    # if return_groups:
    #     return GroupFeatureAttrOutput(attrs_all, lime_exp, group_masks_all, group_attrs_all)
    # else:
    attrs_all = attrs_all.permute(1,0) # (L, N)
    return FeatureAttrOutput(attrs_all, lime_exp)
    # explanation_dict = {w: s for w, s in lime_exp.as_list()}
    
    # attrs = torch.tensor([explanation_dict[token] if token in explanation_dict else 0 for token in tokens])

    # return FeatureAttrOutput(attrs, lime_exp)


import torch
from torch import nn

class WrappedModelMinibatch(nn.Module):
    def __init__(self, model, batch_size=16):
        """
        Wraps a model to handle minibatch processing inside the forward method.

        Args:
            model (nn.Module): The underlying model to wrap.
            batch_size (int): The minibatch size for processing.
        """
        super().__init__()
        self.model = model
        self.batch_size = batch_size

    def forward(self, **kwargs):
        """
        Splits the input into minibatches, processes each minibatch, and combines the outputs.

        Args:
            **kwargs: Keyword arguments containing input tensors.

        Returns:
            torch.Tensor: Combined output logits from all minibatches.
        """
        # Extract the input tensor from kwargs (adjust this to match your model's input structure)
        input_tensor = kwargs["input_ids"]  # Example input key, modify as needed
        other_kwargs = {key: value for key, value in kwargs.items() if key != "input_ids"}

        # Prepare to store outputs
        outputs_list = []

        # Process in minibatches
        for start_idx in range(0, input_tensor.size(0), self.batch_size):
            end_idx = start_idx + self.batch_size

            # Slice the minibatch inputs
            minibatch_input = input_tensor[start_idx:end_idx]
            minibatch_kwargs = {key: value[start_idx:end_idx] for key, value in other_kwargs.items()}

            # Forward pass for the minibatch
            minibatch_output = self.model(input_ids=minibatch_input, **minibatch_kwargs)
            if isinstance(minibatch_output, torch.Tensor):
                logits = minibatch_output
            else:
                logits = minibatch_output.logits
            outputs_list.append(logits.detach())

        # Concatenate all minibatch outputs
        combined_outputs = torch.cat(outputs_list, dim=0)
        return combined_outputs



class LimeTextCls(FeatureAttrMethod):
    def __init__(self, model, tokenizer,
                 LimeTextExplainerKwargs={},
                 explain_instance_kwargs={
                     # Make this however big you need to get every label
                     # this is because the original LIME API is stupid
                     "top_labels" : 1000000,
                     "num_samples" : 500,
                 },
                 batch_size=16):
        super().__init__(model)
        self.tokenizer = tokenizer
        self.LimeTextExplainerKwargs = LimeTextExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs

        self.model_minibatch = WrappedModelMinibatch(model, batch_size=batch_size)

    def forward(self, x, t, return_groups=False):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        N = x.size(0)
        assert x.ndim == 2 and t.ndim in [1, 2] and len(t) == N

        if t.ndim == 1:
            t = t.unsqueeze(1)

        attrs, lime_exps = [], []
        for i in range(N):
            xi, ti = x[i], t[i].cpu().tolist()
            out = explain_text_cls_with_lime(self.model_minibatch, self.tokenizer, xi, ti,
                    LimeTextExplainerKwargs=self.LimeTextExplainerKwargs,
                    explain_instance_kwargs=self.explain_instance_kwargs,
                    return_groups=return_groups)

            attrs.append(out.attributions)
            lime_exps.append(out.explainer_output)

            # if return_groups:
            #     group_masks.append(out.group_masks)
            #     group_attrs.append(out.group_attributions)

        attrs = torch.stack(attrs, dim=0)
        # if return_groups:
        #     group_masks = torch.stack(group_masks, dim=0)
        #     group_attrs = torch.stack(group_attrs, dim=0)
        if attrs.ndim == 3 and attrs.size(-1) == 1:
            attrs = attrs.squeeze(-1)
            # if return_groups:
            #     group_masks = group_masks.squeeze(-1)
            #     group_attrs = group_attrs.squeeze(-1)

        # if return_groups:
        #     return GroupFeatureAttrOutput(attrs, lime_exps, group_masks, group_attrs)
        # else:
        return FeatureAttrOutput(attrs, lime_exps)

        # return FeatureAttrOutput(torch.stack(attrs), lime_exps)