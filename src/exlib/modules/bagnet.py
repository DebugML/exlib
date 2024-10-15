import torch
import torch.nn as nn
from collections import namedtuple
from .libs.bagnets import pytorchnet
from .libs.bagnets.utils import plot_heatmap, generate_heatmap_pytorch
import torchvision.transforms as transforms
from ..explainers.common import get_explanations_in_minibatches, get_binary_masks
import torch.nn.functional as F


AttributionOutputBagNet = namedtuple("AttributionOutputBagNet", [
    "logits",
    "attributions",
    "group_masks",
    "group_attributions"
    ])


class BagNet(nn.Module):
    def __init__(self,
                 model_name='bagnet33',
                 pretrained=True,
                 patchsize=None):
        super().__init__()

        self.model_name = model_name

        self.type_mapping = {
            'bagnet9': (pytorchnet.bagnet9, 9),
            'bagnet17': (pytorchnet.bagnet17, 17),
            'bagnet33': (pytorchnet.bagnet33, 33)
        }

        self.model = self.type_mapping[model_name][0](pretrained=pretrained)
        self.model.eval()
        if patchsize is None:
            self.patchsize = self.type_mapping[model_name][1]
        else:
            self.patchsize = patchsize

    def normalize(self, x):
        x_norm = (x + 1)/2
        x_norm -= torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(x.device)
        x_norm /= torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(x.device)
        return x_norm

    def forward(self, x, t=None, return_groups=True, mini_batch_size=16, **kwargs):
        x = x.clone().detach().requires_grad_()
        x_norm = self.normalize(x)

        if not return_groups:
            logits = self.model(x_norm, return_groups=return_groups)
            group_attributions = None
        else:
            logits, group_attributions = self.model(x_norm, return_groups=return_groups)

        if t is None:
            t = logits.argmax(dim=1)

        if t.ndim == 1:
            t = t[:, None]
        
        if not return_groups:
            logits = self.model(x, return_groups=return_groups)
            if t is None:
                t = logits.argmax(dim=1)

            def get_attr_fn(x, t, model, patchsize):
                x = x.clone().detach().requires_grad_()
                # x = self.normalize(x)

                attrs = []
                for i in range(len(x)):
                    heatmap = generate_heatmap_pytorch(model, x[i:i+1].cpu(), t[i:i+1].cpu(), 
                                patchsize)
                    attrs.append(torch.tensor(heatmap)[None])

                attrs = torch.stack(attrs, 0).to(x.device)
                assert attrs.shape[0] == x.shape[0]
                return attrs
            
            attrs, _ = get_explanations_in_minibatches(x_norm, t, get_attr_fn, mini_batch_size=mini_batch_size, 
                            show_pbar=False, model=self.model, patchsize=self.patchsize)
            attrs = attrs.permute(0, 4, 1, 2, 3)[torch.arange(t.shape[0])[:, None], t].permute(0, 2, 3, 4, 1)

        else:
            attrs = F.interpolate(group_attributions.permute(0, 3, 1, 2),
                    size=(x.shape[-2], x.shape[-1]), mode='nearest').permute(0, 2, 3, 1)[:,None]

            # import pdb; pdb.set_trace()
            group_attributions = group_attributions.permute(
                0, 3, 1, 2)[torch.arange(t.shape[0])[:, None], t].permute(0, 2, 3, 1)
            # import pdb; pdb.set_trace()
            group_attributions = group_attributions / (group_attributions.shape[1] * \
                group_attributions.shape[2])
            # import pdb; pdb.set_trace()
            group_masks = get_binary_masks(x.shape[-2], self.patchsize, 
                                           group_attributions.shape[2]).to(x.device)
            group_masks = group_masks[None].repeat(x.shape[0], 1, 1, 1) # (N, M, H, W)
        
        
        if attrs.ndim == 5 and attrs.size(-1) == 1:
            attrs = attrs.squeeze(-1)
        
        if group_attributions.ndim == 4 and group_attributions.size(-1) == 1:
            group_attributions = group_attributions.squeeze(-1)

        return AttributionOutputBagNet(logits, attrs, group_masks, group_attributions)