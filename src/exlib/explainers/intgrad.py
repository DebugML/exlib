import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .common import *


def intgrad_image_class_loss_fn(y, label):
    N, K = y.shape
    assert len(label) == N
    # Make sure the dtype is right otherwise loss will be like all zeros
    loss = torch.zeros_like(label, dtype=y.dtype)
    for i, l in enumerate(label):
        loss[i] = y[i,l]
    return loss


def intgrad_image_seg_loss_fn(y, label):
    N, K, H, W = y.shape
    assert len(label) == N
    loss = torch.zeros_like(label, dtype=y.dtype)
    for i, l in enumerate(label):
        yi = y[i]
        inds = yi.argmax(dim=0) # Max along the channels
        H = F.one_hot(inds, num_classes=K)  # (H,W,K)
        H = H.permute(2,0,1)   # (K,H,W)
        L = (yi * H).sum()
        loss[i] = L
    return loss


# Do classification-based thing
def explain_image_with_intgrad(x, model, loss_fn,
                               x0 = None,
                               num_steps = 32,
                               progress_bar = False):
    """
    Explain a classification model with Integrated Gradients.
    """
    # Default baseline is zeros
    x0 = torch.zeros_like(x) if x0 is None else x0

    step_size = 1 / num_steps
    intg = torch.zeros_like(x)

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)

    for k in pbar:
        ak = k * step_size
        xk = x0 + ak * (x - x0)
        xk.requires_grad_()
        y = model(xk)

        loss = loss_fn(y)
        loss.sum().backward()
        intg += xk.grad * step_size

    return FeatureAttrOutput(intg, {})


def explain_image_cls_with_intgrad(model, x, label,
                             x0 = None,
                             num_steps = 32,
                             progress_bar = False):
    """
    Explain a classification model with Integrated Gradients.
    """
    assert x.size(0) == len(label)

    # Default baseline is zeros
    x0 = torch.zeros_like(x) if x0 is None else x0

    step_size = 1 / num_steps
    intg = torch.zeros_like(x)

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)
    for k in pbar:
        ak = k * step_size
        xk = x0 + ak * (x - x0)
        xk.requires_grad_()
        y = model(xk)

        loss = 0.0
        for i, l in enumerate(label):
            loss += y[i, l]

        loss.backward()
        intg += xk.grad * step_size

    return FeatureAttrOutput(intg, {})


class IntGradImageCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model, num_steps=32):
        super().__init__(model)
        self.num_steps = num_steps

    def forward(self, x, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        with torch.enable_grad():
            return explain_image_cls_with_intgrad(self.model, x, t, num_steps=self.num_steps, **kwargs)



class IntGradImageSeg(FeatureAttrMethod):
    """ Image segmentation with integrated gradients.
    For this we convert the segmentation model into a classification model.
    """
    def __init__(self, model):
        super().__init__(model, num_steps=32)

        self.cls_model = Seg2ClsWrapper(model)
        self.num_steps = num_steps

    def forward(self, x, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        with torch.enable_grad():
            return explain_image_cls_with_intgrad(self.cls_model, x, t, num_steps=self.num_steps, **kwargs)


def explain_text_cls_with_intgrad(model, x, label,
                             x0 = None,
                             num_steps = 32,
                             progress_bar = False,
                             x_kwargs = {},
                             mask_combine=None):
    """
    Explain a classification model with Integrated Gradients.
    """
    assert x.size(0) == len(label)

    # Default baseline is zeros
    x0 = torch.zeros_like(x) if x0 is None else x0

    step_size = 1 / num_steps
    intg = torch.zeros_like(x).float()

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)
    for k in pbar:
        ak = k * step_size
        
        if mask_combine:
            mask = ak * torch.ones_like(x[0]).float()
            xk = mask_combine(x, mask).squeeze(1)
            xk.requires_grad_()
            y = model(inputs_embeds=xk, **x_kwargs)
        else:
            xk = x0 + ak * (x - x0)
            xk.requires_grad_()
            y = model(xk, **x_kwargs)

        loss = 0.0
        for i, l in enumerate(label):
            loss += y[i, l]

        loss.backward()
        if mask_combine is not None:
            intg += xk.grad.sum(-1) * step_size
        else:
            intg += xk.grad * step_size

    return FeatureAttrOutput(intg, {})


class IntGradTextCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model, mask_combine='default', projection_layer=None, num_steps=32):
        super().__init__(model)
        if mask_combine == 'default':
            def mask_combine(inputs, masks):
                with torch.no_grad():
                    inputs_embed = projection_layer(inputs)
                    mask_embed = projection_layer(torch.tensor([0]).int().to(inputs.device))
                    masked_inputs_embeds = inputs_embed.unsqueeze(1) * masks.unsqueeze(-1) + \
                            mask_embed.view(1,1,1,-1) * (1 - masks.unsqueeze(-1))
                return masked_inputs_embeds
        self.mask_combine = mask_combine
        self.num_steps = num_steps

    def forward(self, x, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        with torch.enable_grad():
            return explain_text_cls_with_intgrad(self.model, x, t, 
                                                 mask_combine=self.mask_combine,
                                                num_steps=self.num_steps,
                                                 **kwargs)