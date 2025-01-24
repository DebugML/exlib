import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from .common import *
from .libs.saliency.text_wrapper import ExtraDimModelWrapper
from .libs.saliency.saliency_zoo import mfaba_cos, mfaba_norm, mfaba_sharp, mfaba_smooth

class MfabaImageCls(FeatureAttrMethod):
    def __init__(self, model, mfaba_type='sharp', mfaba_args={}):
        super().__init__(model)
        self.mfaba_type = mfaba_type
        self.mfaba_args = mfaba_args

    def forward(self, x, t, return_groups=False):
        type_mapping = {
            'sharp': mfaba_sharp,
            'smooth': mfaba_smooth,
            'cos': mfaba_cos,
            'norm': mfaba_norm
        }
        mfaba = type_mapping[self.mfaba_type]
        if t.ndim == 1:
            t = t.unsqueeze(1)
        with torch.enable_grad():

            def get_attr_fn(x, t, model, **kwargs):
                x = x.clone().detach().requires_grad_()
                return torch.tensor(mfaba(model, x, t, **kwargs), device=x.device)

            attributions_all, _ = get_explanations_in_minibatches(x, t, get_attr_fn, mini_batch_size=16, show_pbar=False,
                model=self.model, **self.mfaba_args)
            
            attrs = attributions_all
            if attrs.ndim == 5 and attrs.size(-1) == 1:
                attrs = attrs.squeeze(-1)
            return FeatureAttrOutput(attrs, {})

        

class MfabaTextCls(FeatureAttrMethod):
    def __init__(self, model, projection_layer, mfaba_type='sharp', mfaba_args={}):
        super().__init__(model)
        self.projection_layer = projection_layer
        self.mfaba_type = mfaba_type
        self.mfaba_args = mfaba_args
        self.wrapped_model = ExtraDimModelWrapper(model).to(next(model.parameters()).device)

    def forward(self, x, t, x_kwargs={}, return_groups=False):
        # didn't implement using x_kwargs because it requires 
        # too much manipulation in mfaba
        type_mapping = {
            'sharp': mfaba_sharp,
            'smooth': mfaba_smooth,
            'cos': mfaba_cos,
            'norm': mfaba_norm
        }
        mfaba = type_mapping[self.mfaba_type]
        if t.ndim == 1:
            t = t.unsqueeze(1)
        x_embed = self.projection_layer(x)
        with torch.enable_grad():

            def get_attr_fn(x_embed, t, x_kwargs, model, **kwargs):
                x_embed = x_embed.clone().detach().requires_grad_().unsqueeze(1)
                mfaba_results = torch.tensor(mfaba(model, x_embed, t, **kwargs), device=x.device)
                return mfaba_results.squeeze(1)
                
            attributions_all, _ = get_explanations_in_minibatches_text(x_embed, t, 
                get_attr_fn, mini_batch_size=16, x_kwargs=x_kwargs,
                show_pbar=False,
                model=self.wrapped_model, **self.mfaba_args)
                        
            attrs = attributions_all
            if attrs.ndim == 3 and attrs.size(-1) == 1:
                attrs = attrs.squeeze(-1)
            return FeatureAttrOutput(attrs, {})