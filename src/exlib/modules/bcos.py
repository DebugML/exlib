import torch
import torch.nn as nn
from collections import namedtuple
from bcos.data.transforms import AddInverse
from ..explainers.common import get_explanations_in_minibatches

AttributionOutputBCos = namedtuple("AttributionOutputBCos", [
    "preds",
    "attributions",
    ])


class BCos(nn.Module):
    def __init__(self,
                 model_name='simple_vit_b_patch16_224'):
        super(BCos, self).__init__()

        self.model = torch.hub.load('B-cos/B-cos-v2', model_name, pretrained=True)

    def preprocess(self, x):
        x = (x + 1) / 2
        return AddInverse()(x)

    def forward(self, x, t=None, return_groups=False, **kwargs):
        def get_attr_fn(x, t, model, preprocess):
            x = x.clone().detach().requires_grad_()
            in_tensor = preprocess(x)
            expln = model.explain(in_tensor, idx=t)
            attrs = torch.tensor(expln['explanation'], device=x.device).permute(2,0,1)
            # print('attrs', attrs.shape)
            return attrs[None, :3], torch.tensor([expln['prediction']], device=x.device)

        attrs, preds = get_explanations_in_minibatches(x, t, get_attr_fn, mini_batch_size=1, 
                        show_pbar=False, model=self.model, preprocess=self.preprocess)

        if attrs.ndim == 5 and attrs.size(-1) == 1:
            attrs = attrs.squeeze(-1)

        return AttributionOutputBCos(
            preds,
            attrs)