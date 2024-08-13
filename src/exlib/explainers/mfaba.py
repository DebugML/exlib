import copy
import torch
import torch.nn.functional as F
import numpy as np
from .common import *
from .libs.saliency.saliency_zoo import mfaba_cos, mfaba_norm, mfaba_sharp, mfaba_smooth

class MfabaImageCls(FeatureAttrMethod):
    def __init__(self, model, mfaba_type='sharp'):
        super().__init__(model)
        self.mfaba_type = mfaba_type

    def forward(self, x, t, **kwargs):
        type_mapping = {
            'sharp': mfaba_sharp,
            'smooth': mfaba_smooth,
            'cos': mfaba_cos,
            'norm': mfaba_norm
        }
        mfaba = type_mapping[self.mfaba_type]
        with torch.enable_grad():
            return FeatureAttrOutput(torch.tensor(mfaba(self.model, x, t, **kwargs)).to(x.device), {})