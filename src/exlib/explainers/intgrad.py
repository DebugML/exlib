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
                             progress_bar = False,
                             return_groups=False,
                             mini_batch_size=16):
    """
    Explain a classification model with Integrated Gradients.
    """
    assert x.size(0) == len(label)
    if label.ndim == 1:
        label = label.unsqueeze(1)

    # Default baseline is zeros
    x0 = torch.zeros_like(x) if x0 is None else x0

    step_size = 1 / num_steps

    def get_attr_fn(x, t, model, step_size):
        x = x.clone().detach().requires_grad_()
        y = model(x)
        loss = y.gather(1, t[:,None])
        loss.sum().backward()
        return x.grad * step_size

    intg = torch.zeros(x.size(0), x.size(1), x.size(2), x.size(3), label.size(1), 
                        device=x.device, dtype=x.dtype)

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)

    for k in pbar:
        ak = k * step_size
        xk = x0 + ak * (x - x0)
        xk.requires_grad_()

        attr_curr, _ = get_explanations_in_minibatches(xk, label, get_attr_fn, mini_batch_size, 
                show_pbar=progress_bar, model=model, step_size=step_size)
        intg += attr_curr

    attrs = intg
    if attrs.ndim == 5 and attrs.size(-1) == 1:
        attrs = attrs.squeeze(-1)

    return FeatureAttrOutput(attrs, {})


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
                             mask_combine=None,
                             mini_batch_size=16):
    """
    Explain a classification model with Integrated Gradients.
    """
    assert x.size(0) == len(label)
    if label.ndim == 1:
        label = label.unsqueeze(1)

    # Default baseline is zeros
    x0 = torch.zeros_like(x) if x0 is None else x0

    step_size = 1 / num_steps

    def get_attr_fn(x, t, x_kwargs, model, step_size, do_mask_combine=False):
        x = x.clone().detach().requires_grad_()
        if mask_combine:
            y = model(inputs_embeds=x, **x_kwargs)
        else:
            y = model(x, **x_kwargs)
        loss = y.gather(1, t[:,None])
        loss.sum().backward()
        return x.grad * step_size

    intg = torch.zeros(x.size(0), x.size(1), label.size(1), device=x.device, dtype=torch.float)

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)
    for k in pbar:
        ak = k * step_size
        
        if mask_combine:
            mask = ak * torch.ones_like(x).float()
            xk = mask_combine(x, mask).squeeze(1)
            xk.requires_grad_()
        else:
            xk = x0 + ak * (x - x0)
            xk.requires_grad_()
        
        attr_curr, _ = get_explanations_in_minibatches_text(xk, label, 
                lambda x, t, x_kwargs, model, step_size: get_attr_fn(x, t, x_kwargs, model, step_size, 
                    do_mask_combine=mask_combine), mini_batch_size, x_kwargs=x_kwargs,
                show_pbar=progress_bar, model=model, step_size=step_size)

        intg += attr_curr

    attrs = intg
    if attrs.ndim == 3 and attrs.size(-1) == 1:
        attrs = attrs.squeeze(-1)

    return FeatureAttrOutput(attrs, {})

class IntGradTextCls(FeatureAttrMethod):
    """ Text classification with integrated gradients
    """
    def __init__(self, model, mask_combine='default', projection_layer=None, num_steps=32):
        super().__init__(model)
        if mask_combine == 'default':
            def mask_combine(inputs: torch.LongTensor, masks: torch.FloatTensor):
                with torch.no_grad():
                    inputs_embed = projection_layer(inputs) # (bsz, l, d)
                    bsz, l, d = inputs_embed.shape
                    mask_embed = projection_layer(torch.tensor([[0]]).int().to(inputs.device)) # (1, 1, d)
                    masked_inputs_embeds = inputs_embed * masks.view(bsz,l,1) + mask_embed.view(1,1,d) * (1 - masks.view(bsz,l,1))
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