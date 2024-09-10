import torch
import torch.nn as nn
from collections import namedtuple
import bagnets.pytorchnet
from bagnets.utils import plot_heatmap, generate_heatmap_pytorch
import torchvision.transforms as transforms
from ..explainers.common import get_explanations_in_minibatches


AttributionOutputBagNet = namedtuple("AttributionOutputBagNet", [
    "logits",
    "attributions"
    ])


class BagNet(nn.Module):
    def __init__(self,
                 model_name='bagnet33',
                 pretrained=True,
                 patchsize=None):
        super().__init__()

        self.model_name = model_name

        self.type_mapping = {
            'bagnet9': (bagnets.pytorchnet.bagnet9, 9),
            'bagnet17': (bagnets.pytorchnet.bagnet17, 17),
            'bagnet33': (bagnets.pytorchnet.bagnet33, 33)
        }

        self.model = self.type_mapping[model_name][0](pretrained=pretrained)
        if patchsize is None:
            self.patchsize = self.type_mapping[model_name][1]
        else:
            self.patchsize = patchsize

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

    def forward(self, x, t=None, return_groups=False, mini_batch_size=16, **kwargs):
        def get_attr_fn(x, t, model, patchsize):
            x = x.clone().detach().requires_grad_()
            x = self.normalize(x)
            outputs = model(x)
            if t is None:
                t = outputs.argmax(dim=1)

            # attrs = generate_heatmap_pytorch(model, x.cpu(), t.cpu(), patchsize)
            # print('attrs', attrs.shape)
            # return torch.tensor(attrs)[None], outputs
            attrs = []
            for i in range(len(x)):
                heatmap = generate_heatmap_pytorch(model, x[i:i+1].cpu(), t[i:i+1].cpu(), 
                            patchsize)
                attrs.append(torch.tensor(heatmap)[None])

            attrs = torch.stack(attrs, 0).to(x.device)
            assert attrs.shape[0] == x.shape[0]
            return attrs, outputs
        
        attrs, preds = get_explanations_in_minibatches(x, t, get_attr_fn, mini_batch_size=mini_batch_size, 
                        show_pbar=False, model=self.model, patchsize=self.patchsize)

        if attrs.ndim == 5 and attrs.size(-1) == 1:
            attrs = attrs.squeeze(-1)

        return AttributionOutputBagNet(preds, attrs)