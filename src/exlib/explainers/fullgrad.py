from exlib.explainers.common import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .libs.fullgrad.saliency.fullgrad import FullGrad
from .libs.fullgrad.saliency.fullgradvit import FullGradViT

def explain_image_cls_with_fullgrad(fullgrad, x, label, model_type='vit'):
    """
    Explain a classification model with Integrated Gradients.
    """
    assert x.size(0) == len(label)
    if label.ndim == 1:
        label = label.unsqueeze(1)

    # Obtain saliency maps
    attrs = []
    for i in range(x.size(0)):
        attrs_curr = []
        for l in tqdm(label[i]):
            saliency_map = fullgrad.saliency(x[i:i+1], l)
            # import pdb; pdb.set_trace()
            attrs_curr.append(saliency_map)
        attrs_curr = torch.stack(attrs_curr, dim=-1)
        attrs.append(attrs_curr)
    attrs = torch.cat(attrs, dim=0)
    # saliency_map = fullgrad.saliency(x, label)
    # import pdb; pdb.set_trace()

    return FeatureAttrOutput(saliency_map, {})


class FullGradImageCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model, im_size=(3, 224, 224), model_type='vit', check_completeness=False):
        super().__init__(model)
        print('init fullgrad')
        if model_type == 'vit':
            self.fullgrad = FullGradViT(model, im_size=im_size, check_completeness=check_completeness)
        else:
            self.fullgrad = FullGrad(model, im_size=im_size, check_completeness=check_completeness)
        print('init fullgrad done')

    def forward(self, x, t, return_groups=False, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        print('start fullgrad')

        with torch.enable_grad():
            return explain_image_cls_with_fullgrad(self.fullgrad, x, t, **kwargs)
