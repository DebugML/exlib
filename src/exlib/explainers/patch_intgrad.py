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



def explain_cls_with_patchintgrad(model, x, label,
                             x0 = None,
                             num_steps = 32,
                             progress_bar = False,
                             explain_instance_kwargs={}):
    """
    Explain a classification model with Integrated Gradients.
    """
    segmentation_fn = explain_instance_kwargs.get('segmentation_fn')

    if segmentation_fn is None:
        segments = patch_segment(x,patch_height=56,patch_width=56,dtype="torch")

    else:
        segments = segmentation_fn(x) #Look into providing arguments to this

    n_segments = segments.max()



    # Default baseline is zeros
    x0 = torch.zeros_like(x) if x0 is None else x0

    step_size = 1 / num_steps
    intg = torch.zeros((n_segments,))

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)
    for k in pbar:
        ak = k * step_size
        patch_alphas = [torch.tensor(ak,requires_grad=True).to(x.device) for j in range(n_segments)]
        for i in range(n_segments):
            patch_alphas[i].retain_grad()

        xk = x0.clone().detach()

        for i in range(n_segments):
            mask = (segments == i).to(x.device)   #creating mask
            xk += x0 + patch_alphas[i] * (x - x0)*mask

        # xk.requires_grad_()
        xk = xk.unsqueeze(0)
        y = model(xk)

        loss = 0.0
        for i, l in enumerate(label): #single label so probably don't need to do this
            loss += y[i, l]

        loss.backward()
        # intg += xk.grad * step_size

        intg += torch.tensor([patch_alphas[j].grad*step_size for j in range(n_segments)])

    intg_fin = torch.zeros_like(segments).to(x.device).float()

    for i in range(n_segments):
        mask = (segments == i).to(x.device)
        intg_fin += mask*intg[i]


    return FeatureAttrOutput(intg_fin, {})


class PatchIntGradImageCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model):
        super().__init__(model)

    def forward(self, x, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        # with torch.enable_grad():
        #     return explain_cls_with_intgrad(self.model, x, t, **kwargs)

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        N = x.size(0)
        assert x.ndim == 4 and t.ndim == 1 and len(t) == N

        attrs, intgrad_exps = [], []
        for i in range(N):
            xi, ti = x[i], t[i].cpu().item()
            out = explain_cls_with_patchintgrad(self.model, xi, [ti],**kwargs)

            attrs.append(out.attributions)
            intgrad_exps.append(out.explainer_output)



        return FeatureAttrOutput(torch.stack(attrs), intgrad_exps)
