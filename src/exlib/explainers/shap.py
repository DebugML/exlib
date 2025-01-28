import copy
import torch
import torch.nn.functional as F
import shap
import numpy as np
from .common import *


def explain_image_cls_with_shap(model, x, t, mask_value, shap_explainer_kwargs):
    assert len(x) == len(t)
    device = next(model.parameters()).device

    def f(x_np):
        with torch.no_grad():
            pred = model(np_to_torch_img(x_np).to(device))
            return pred.detach().cpu().numpy()

    # By default the Partition explainer is used for all partition explainer
    x_np = torch_img_to_np(x.cpu())
    masker = shap.maskers.Image(mask_value, x_np[0].shape)
    explainer = shap.Explainer(f, masker, **shap_explainer_kwargs)

    shap_outs = []
    shap_values = []
    for xi, ti in zip(x_np, t):
        if isinstance(ti, torch.Tensor):
            if len(ti.shape) == 0:
                ti = [ti.cpu().item()]
            else:
                ti = ti.cpu().numpy().tolist()
        else:
            if isinstance(ti, int):
                ti = [ti]
        out = explainer(np.expand_dims(xi, axis=0), outputs=ti)
        svs = torch.from_numpy(out.values).to(x.device) # (1,H,W,C,1)
        shap_outs.append(out)
        shap_values.append(svs[0].permute(2,0,1,3)) # (C,H,W)
    shap_values = torch.stack(shap_values)
    attrs = shap_values
    if attrs.ndim == 5 and attrs.size(-1) == 1:
        attrs = attrs.squeeze(-1)
    return FeatureAttrOutput(attrs, shap_outs)


class ShapImageCls(FeatureAttrMethod):
    def __init__(self, model, mask_value=0, shap_explainer_kwargs={}):
        super().__init__(model)
        self.mask_value = mask_value
        self.shap_explainer_kwargs = shap_explainer_kwargs

    def forward(self, x, t, **kwargs):
        return explain_image_cls_with_shap(self.model, x, t, self.mask_value, self.shap_explainer_kwargs)


class ShapImageSeg(FeatureAttrMethod):
    def __init__(self, model, mask_value=0, shap_explainer_kwargs={}):
        super().__init__(model)
        self.mask_value = mask_value
        self.shap_explainer_kwargs = shap_explainer_kwargs
        self.cls_model = Seg2ClsWrapper(model)

    def forward(self, x, t, **kwargs):
        return explain_image_cls_with_shap(self.cls_model, x, t, self.mask_value, self.shap_explainer_kwargs)

def explain_text_cls_with_shap(model, tokenizer, x, t, mask_value, shap_explainer_kwargs, pad_value=0, special_tokens=[]):
    # assert len(x[list(x.keys())[0]]) == len(t)
    assert len(x) == len(t)
    device = next(model.parameters()).device

    input_ids = x

    inputs_raw = []
    for input_ids_i in input_ids:
        input_raw = tokenizer.decode(input_ids_i)
        for st in special_tokens:
            input_raw = input_raw.replace(st, '')
        inputs_raw.append(input_raw)


    # inputs_raw = [tokenizer.decode(input_ids_i) for input_ids_i in input_ids]

    def f(x_str):
        with torch.no_grad():
            inputs = tokenizer(x_str.tolist(), 
                                padding='max_length', 
                                truncation=True, 
                                max_length=512,
                                add_special_tokens=False)
            
            inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
            pred = model(**inputs)

            return pred.detach().cpu().numpy()

    explainer = shap.Explainer(f, tokenizer, **shap_explainer_kwargs)
    # shap_outs = explainer(x)

    def pad_tensor_to_length(tensor, target_length=512, pad_value=0):
        """Pad tensor with pad_value up to target_length."""
        pad_length = target_length - tensor.size(-1)
        # print('tensor:', tensor.shape)
        # print('target_length:', target_length)
        # print('pad_length:', pad_length)
        return F.pad(tensor, (0, pad_length), 'constant', pad_value)
    
    # import pdb; pdb.set_trace()
    shap_outs = []
    shap_values = []
    for xi, ti in zip(inputs_raw, t):
        if isinstance(ti, torch.Tensor):
            if len(ti.shape) == 0:
                ti = [ti.cpu().item()]
            else:
                ti = ti.cpu().numpy().tolist()
        else:
            if isinstance(ti, int):
                ti = [ti]
        # out = explainer(np.expand_dims(xi, axis=0), outputs=ti)
        # import pdb; pdb.set_trace()
        # print('xi:', xi)
        out = explainer([xi], outputs=ti)
        # print('out.values:', out.values.shape)
        svs = torch.from_numpy(out.values).to(device) # (1, L, T)
        shap_outs.append(out)
        
        padded_svs = [pad_tensor_to_length(tensor.transpose(1,0), target_length=input_ids.shape[1], pad_value=pad_value) for tensor in svs]
        # print('padded_svs:', padded_svs[0].shape)

        shap_values.append(padded_svs[0].transpose(1,0)) # (N, T, L) -> (T, N, L)
    shap_values = torch.stack(shap_values)
    # import pdb; pdb.set_trace()
    attrs = shap_values
    # print(attrs.shape)
    if attrs.ndim == 3 and attrs.size(-1) == 1:
        attrs = attrs.squeeze(-1)
    # print('attrs:', attrs.shape)
    return FeatureAttrOutput(attrs, shap_outs)
    

    # shap_values = []
    # svs = [torch.tensor(sv[:,t[sv_i]]) 
    #        for sv_i, sv in enumerate(shap_outs.values)]
    # def pad_tensor_to_length(tensor, target_length=512, pad_value=0):
    #     """Pad tensor with pad_value up to target_length."""
    #     pad_length = target_length - tensor.size(0)
    #     return F.pad(tensor, (0, pad_length), 'constant', pad_value)
    # padded_svs = [pad_tensor_to_length(tensor) for tensor in svs]
    # shap_values = torch.stack(padded_svs, dim=0)
    
    # return FeatureAttrOutput(shap_values, shap_outs)


class ShapTextCls(FeatureAttrMethod):
    def __init__(self, model, tokenizer, mask_value=0, shap_explainer_kwargs={}, pad_value=0, special_tokens=['[CLS]', '[SEP]']):
        super().__init__(model)
        self.tokenizer = tokenizer
        self.mask_value = mask_value
        self.shap_explainer_kwargs = shap_explainer_kwargs
        self.pad_value = pad_value
        self.special_tokens = special_tokens

    def forward(self, x: torch.LongTensor, t, **kwargs):
        return explain_text_cls_with_shap(self.model, self.tokenizer,
                                          x, t, self.mask_value, self.shap_explainer_kwargs, self.pad_value, self.special_tokens)

"""
Usage: (attention: special tokens are different for BERT and Roberta)

BERT:
from transformers import BertTokenizer, BertForSequenceClassification
from exlib.explainers.shap import ShapTextCls

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# input_ids: torch.LongTensor of shape (N, L)
# pred: torch.LongTensor of shape (N,)
# N: number of samples
# L: number of tokens

# For BERT:
explainer = ShapTextCls(model, tokenizer, pad_value=0, special_tokens=['[CLS]', '[SEP]'])

expln = explainer(input_ids, pred)

# RoBERTa:
explainer = ShapTextCls(model, tokenizer, pad_value=1, special_tokens=['<s>', '</s>'])
    
expln = explainer(input_ids, pred)
"""

