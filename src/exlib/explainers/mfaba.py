import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from .common import *
from .libs.saliency.saliency_zoo import mfaba_cos, mfaba_norm, mfaba_sharp, mfaba_smooth

class MfabaImageCls(FeatureAttrMethod):
    def __init__(self, model, mfaba_type='sharp'):
        super().__init__(model)
        self.mfaba_type = mfaba_type

    def forward(self, x, t, return_groups=False, **kwargs):
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
            # import time
            # start = time.time()
            attributions_all = torch.zeros((x.size(0), x.size(1), x.size(2), x.size(3), t.size(1)))
            for t_idx in tqdm(range(t.size(0))):
                for i in range(t.size(1)):
                    t_curr = t[t_idx, i]
                    t_curr = t_curr.unsqueeze(0)
                    x_curr = x[t_idx].unsqueeze(0)
                    attributions_all[t_idx, :, :, :, i] = torch.tensor(mfaba(self.model, x_curr, t_curr, **kwargs))
                    # attributions_all[:, :, :, :, i] = torch.tensor(mfaba(self.model, x, t_curr, **kwargs))
                    # if i == 0 and t_idx == 0:
                    #     attributions = torch.tensor(mfaba(self.model, x, t_curr, **kwargs))
                    # else:
                    #     attributions = torch.cat((attributions, torch.tensor(mfaba(self.model, x, t_curr, **kwargs))), dim=0)
            attributions_all1 = attributions_all.clone()
            # print('Time taken', time.time() - start)

            # # 2nd way, do them together
            # start = time.time()
            # attributions_all = torch.zeros((x.size(0), x.size(1), x.size(2), x.size(3), t.size(1)))
            # for i in tqdm(range(t.size(1))):
            #     t_curr = t[:, i]
            #     t_curr = t_curr #.unsqueeze(1)
            #     # print('t_curr', t_curr.size())
            #     x_curr = x #.unsqueeze(0)
            #     # x_curr = x.expand(t_curr.size(0), -1, -1, -1).clone().detach().requires_grad_(True)
            #     # print('x_curr', x_curr.size())
            #     attributions_all[:, :, :, :, i] = torch.tensor(mfaba(self.model, x_curr, t_curr, **kwargs))
            #     # if i == 0:
            #     #     attributions = torch.tensor(mfaba(self.model, x, t_curr, **kwargs))
            #     # else:
            #     #     attributions = torch.cat((attributions, torch.tensor(mfaba(self.model, x, t_curr, **kwargs))), dim=0)
            # # attributions = torch.tensor(mfaba(self.model, x, t, **kwargs))

            # # import pdb; pdb.set_trace()
            # print('Time taken', time.time() - start)

            # print(torch.allclose(attributions_all, attributions_all1, atol=1e-6))
            # print('attributions_all', attributions_all.size())
            
            attrs = attributions_all
            if attrs.ndim == 5 and attrs.size(-1) == 1:
                attrs = attrs.squeeze(-1)
            return FeatureAttrOutput(attrs, {})

            # return FeatureAttrOutput(torch.tensor(mfaba(self.model, x, t, **kwargs)).to(x.device), {})