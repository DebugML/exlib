from exlib.explainers.common import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .libs.fullgrad.saliency.fullgrad import FullGrad
from .libs.fullgrad.saliency.fullgradvit import FullGradViT
from exlib.explainers.libs.fullgrad.saliency.fullgradbert import FullGradBERT
from exlib.explainers.libs.saliency.text_wrapper import ExtraDimModelWrapper

def explain_image_cls_with_fullgrad(fullgrad, x, label, model_type='vit'):
    """
    Explain a classification model with Integrated Gradients.
    """
    assert x.size(0) == len(label)
    if label.ndim == 1:
        label = label.unsqueeze(1)

    # Obtain saliency maps
    def get_attr_fn(x, t, fullgrad):
        x = x.clone().detach().requires_grad_()
        return fullgrad.saliency(x, t)

    attrs, _ = get_explanations_in_minibatches(x, label, get_attr_fn, mini_batch_size=16, show_pbar=False,
        fullgrad=fullgrad)

    if attrs.ndim == 5 and attrs.size(-1) == 1:
        attrs = attrs.squeeze(-1)

    return FeatureAttrOutput(attrs, {})


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

def explain_text_cls_with_fullgrad(fullgrad, x, label, model_type='vit'):
    """
    Explain a classification model with Integrated Gradients.
    """
    assert x.size(0) == len(label)
    if label.ndim == 1:
        label = label.unsqueeze(1)

    # Obtain saliency maps
    def get_attr_fn(x, t, fullgrad):
        x = x.clone().detach().requires_grad_()
        return fullgrad.saliency(x, t)

    attrs, _ = get_explanations_in_minibatches(x, label, get_attr_fn, mini_batch_size=16, show_pbar=False,
        fullgrad=fullgrad)
    
    # print('attrs', attrs.shape)
    
    if attrs.ndim == 5 and attrs.size(-1) == 1:
        attrs = attrs.squeeze(-1)
    attrs = attrs.squeeze(1).sum(2)
    # return FeatureAttrOutput(attrs, {})

    return FeatureAttrOutput(attrs, {})

class FullGradTextCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model, projection_layer, im_size=(1, 512, 768), model_type='bert', check_completeness=False):
        super().__init__(model)
        print('init fullgrad')
        # model_type = bert
        self.wrapped_model = ExtraDimModelWrapper(model).to(next(model.parameters()).device)
        self.fullgrad = FullGradBERT(self.wrapped_model, im_size=im_size, check_completeness=check_completeness)
        self.projection_layer = projection_layer
        
        print('init fullgrad done')

    def forward(self, x, t, return_groups=False, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        x_embed = self.projection_layer(x).unsqueeze(1)
        with torch.enable_grad():
            return explain_text_cls_with_fullgrad(self.fullgrad, x_embed, t, **kwargs)