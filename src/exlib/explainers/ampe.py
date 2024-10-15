import copy
import torch
import torch.nn.functional as F
import numpy as np
from .common import *
from .libs.saliency.saliency_zoo import ampe
from tqdm.auto import tqdm


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

    def forward(self, x, t, return_groups=False, **kwargs):
        assert len(x) == len(t)

        if t.ndim == 1:
            t = t.unsqueeze(1)

        with torch.enable_grad():
            
            def get_attr_fn(x, t, model, **kwargs):
                x = x.clone().detach().requires_grad_()
                return torch.tensor(ampe(model, x, t, **kwargs), device=x.device)
            
            attributions_all, _ = get_explanations_in_minibatches(x, t, get_attr_fn, mini_batch_size=16, show_pbar=False, 
                model=self.model, data_min=self.data_min, data_max=self.data_max, epsilon=self.epsilon,
                N=self.N, num_steps=self.num_steps, use_sign=self.use_sign, 
                use_softmax=self.use_softmax, verbose=self.verbose)

            # actual correct one:
            # # print('ampe_attr', ampe_attr.shape)
            # attributions_all1 = attributions_all.clone()

            # attributions_all = torch.zeros((x.size(0), x.size(1), x.size(2), x.size(3), t.size(1)), device=x.device)

            # for t_idx in range(t.size(0)):
            #     for i in tqdm(range(t.size(1))):
            #         t_curr = t[t_idx, i]
            #         t_curr = t_curr.unsqueeze(0)
            #         x_curr = x[t_idx].unsqueeze(0)
            #         ampe_attr = ampe(self.model, x_curr, t_curr, self.data_min, self.data_max, self.epsilon, 
            #                     self.N, self.num_steps, self.use_sign, self.use_softmax, self.verbose)
            #         attributions_all[t_idx, :, :, :, i] = torch.tensor(ampe_attr).to(x.device) #torch.tensor(mfaba(self.model, x_curr, t_curr, **kwargs))
            
            
            attrs = attributions_all

            # print(torch.allclose(attributions_all1, attributions_all))
            # import pdb; pdb.set_trace()
            if attrs.ndim == 5 and attrs.size(-1) == 1:
                attrs = attrs.squeeze(-1)
            return FeatureAttrOutput(attrs, {})