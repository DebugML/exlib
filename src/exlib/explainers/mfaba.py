import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from .common import *
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
        # import pdb; pdb.set_trace()
        if t.ndim == 1:
            t = t.unsqueeze(1)
        # print('x', x.size())
        # print('t', t.size())
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

            # return FeatureAttrOutput(torch.tensor(mfaba(self.model, x, t, **kwargs)).to(x.device), {})