import copy
import torch
import torch.nn.functional as F
import numpy as np
from .common import *
from .libs.saliency.saliency_zoo import ampe


class AmpeImageCls(FeatureAttrMethod):
    def __init__(self, model, data_min=0, data_max=1, epsilon=16,N=20,num_steps=10, use_sign=True, use_softmax=True, verbose=False):
        super().__init__(model)
        self.data_min = data_min
        self.data_max = data_max
        self.epsilon = epsilon
        self.N = N
        self.num_steps = num_steps
        self.use_sign = use_sign
        self.use_softmax = use_softmax
        self.verbose = verbose

    def forward(self, x, t, **kwargs):
        assert len(x) == len(t)

        with torch.enable_grad():
            attrs = []
            for i in range(len(x)):
                ampe_attr = ampe(self.model, x[i:i+1], t[i:i+1], self.data_min, self.data_max, self.epsilon, 
                                self.N, self.num_steps, self.use_sign, self.use_softmax, self.verbose)
                attrs.append(torch.tensor(ampe_attr).to(x.device))
        return FeatureAttrOutput(torch.cat(attrs, 0), {})