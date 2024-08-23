import torch
import torch.nn as nn
from collections import namedtuple
from bcos.data.transforms import AddInverse


AttributionOutputBCos = namedtuple("AttributionOutputBCos", [
    "preds",
    "attributions", 
    "explainer_output"
    ])


class BCos(nn.Module):
    def __init__(self,
                 model_name='simple_vit_b_patch16_224'):
        super(BCos, self).__init__()

        self.model = torch.hub.load('B-cos/B-cos-v2', model_name, pretrained=True)

    def prepare_data(self, x):
        return AddInverse()(x)

    def forward(self, x, t=None):
        attrs = []
        preds = []
        for i in range(len(x)):
            images = x[i:i+1].clone().requires_grad_()
            in_tensor = self.prepare_data(images)

            if t is None:
                idx = None
            else:
                idx = t[i:i+1]
            expln = self.model.explain(in_tensor.to(x.device), idx=idx)
            attr = torch.tensor(expln['explanation']).permute(2,0,1)
            attrs.append(attr)
            preds.append(expln['prediction'])
        return AttributionOutputBCos(
            torch.tensor(preds).to(x.device),
            torch.stack(attrs), 
            expln)