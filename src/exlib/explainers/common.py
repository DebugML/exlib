from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FamWrapper(nn.Module):
    """ Wrap a model with a pre/post processing function
    """
    def __init__(self, model, preprocessor=None, postprocessor=None):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def forward(self, x):
        if self.preprocessor:
            x = self.preprocessor(x)

        y = self.model(x)

        if self.postprocessor:
            y = self.postprocessor(y)

        return y


"""
    Explainer output format:
    - attributions: (N, C, H, W) or (N, C, H, W, T) or (N, 1, H, W) or (N, 1, H, W, T)
    - group_masks: (N, C, H, W) or (N, C, H, W, T) or (N, 1, H, W) or (N, 1, H, W, T)
    - group_attributions: (N, M) or (N, M, T)
"""
FeatureAttrOutput = namedtuple("FeatureAttrOutput", ["attributions", "explainer_output"])
GroupFeatureAttrOutput = namedtuple("GroupFeatureAttrOutput", 
    ["attributions", "explainer_output", "group_masks", "group_attributions"]
    )


class FeatureAttrMethod(nn.Module): 
    """ Explaination methods that create feature attributions should follow 
    this signature. """
    def __init__(self, model): 
        super().__init__() 
        self.model = model

    def forward(self, x, t, return_groups=False, **kwargs):
        raise NotImplementedError()


class Seg2ClsWrapper(nn.Module):
    """ Simple wrapper for converting to be classification compatible.
    We are assuming

        (N,C,H,W) --[cls model]--> (N,K)
        (N,C,H,W) --[seg model]--> (N,K,H,W)

        (N,C,H,W) --[wrapd seg]--> (N,K)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)    # (N,K,H,W)
        N, K, H, W = y.shape
        A = y.argmax(dim=1)             # (N,H,W)
        A = F.one_hot(A, num_classes=K) # (N,H,W,K)
        A = A.permute(0,3,1,2)          # (N,K,H,W)
        C = (A * y).sum(dim=(2,3))      # (N,K)
        return C


def patch_segmenter(image, sz=(8,8), return_pt=False): 
    """ Creates a grid of size sz for rectangular patches. 
    Adheres to the sk-image segmenter signature. """
    shape = image.shape
    x = torch.from_numpy(image)
    idx = torch.arange(sz[0]*sz[1]).view(1,1,*sz).float()
    segments = F.interpolate(idx, size=x.size()[:2], mode='nearest').long()
    segments = segments[0,0]
    if not return_pt:
        segments = segments.numpy()
    return segments


def patch_segment(image, patch_height=8, patch_width=8,permute = None,dtype = "torch"):
    #Input: C,H,W
    if permute is not None:
        image = np.transpose(image,permute)

    channels, image_height, image_width = image.shape
    if image_height % patch_height != 0 or image_width % patch_width != 0:
        print("patch height and width need to perfectly divide image")
        raise ValueError

    row_indices = torch.arange(image_height)
    column_indices = torch.arange(image_width)

    row_factors = row_indices // patch_height
    column_factors = column_indices // patch_width

    row_factor_matrix = row_factors[:, None]
    column_factor_matrix = column_factors[None, :]

    segment = column_factor_matrix * (image_height // patch_height) + row_factor_matrix
    if dtype == "torch":
        return segment.int()
    elif dtype == "numpy":
        return segment.numpy()
    else:
        raise ValueError


def torch_img_to_np(x): 
    if x.dim() == 4: 
        return x.permute(0,2,3,1).numpy()
    elif x.dim() == 3: 
        return x.permute(1,2,0).numpy()
    else: 
        raise ValueError("Image tensor doesn't have 3 or 4 dimensions")

def np_to_torch_img(x_np):
    x = torch.from_numpy(x_np) 
    if x.dim() == 4: 
        return x.permute(0,3,1,2)
    elif x.dim() == 3: 
        return x.permute(2,0,1)
    else: 
        raise ValueError("Image array doesn't have 3 or 4 dimensions")

def convert_idx_masks_to_bool(masks):
    """
    input: masks (1, img_dim1, img_dim2)
    output: masks_bool (num_masks, img_dim1, img_dim2)
    """
    unique_idxs = torch.sort(torch.unique(masks)).values
    idxs = unique_idxs.view(-1, 1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[1], 
                                     masks.shape[2])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool


def get_explanations_in_minibatches(x, t, get_attr_fn, mini_batch_size, show_pbar=False, **kwargs):
    """
    Get explanations in minibatches.
    Make a general method that takes in an operation with x, labels (in batches) and other kwargs
    """
    assert x.size(0) == len(t)
    if t.ndim == 1:
        t = t.unsqueeze(1)

    attrs = torch.zeros(x.size(0), t.size(1), x.size(1), x.size(2), x.size(3),
                        device=x.device, dtype=x.dtype)
    attrs_shape = attrs.shape
    attrs = attrs.flatten(0, 1)
    x_expand = x[:,None].expand(x.shape[0], t.shape[-1], x.shape[1], x.shape[2], x.shape[3])
    x_expand_shape = x_expand.shape
    x_expand = x_expand.flatten(0, 1)
    t_expand = t.flatten()

    pbar = range(0, x_expand.shape[0], mini_batch_size)
    if show_pbar:
        pbar = tqdm(pbar)

    preds = []
    for i in pbar:
        x_batch = x_expand[i:i+mini_batch_size].clone().detach().requires_grad_()
        t_batch = t_expand[i:i+mini_batch_size]
        attrs_output = get_attr_fn(x_batch, t_batch, **kwargs) # (N, C, H, W) same as x_batch
        if type(attrs_output) == tuple and len(attrs_output) == 2:
            attrs_i, pred_i = attrs_output
            preds.append(pred_i)
        else:
            attrs_i = attrs_output
        attrs[i:i+mini_batch_size] = attrs_i
        
    if len(preds) > 0:
        preds = torch.cat(preds)

    attrs = attrs.view(attrs_shape).permute(0, 2, 3, 4, 1).contiguous()

    if attrs.size(1) != 1:
        if (attrs[:,0] == attrs[:,1]).all() and (attrs[:,0] == attrs[:,2]).all():
            attrs = attrs[:,0:1]

    return attrs, preds


def get_binary_masks(h, p, n):
    """
    h: height of the image 224
    p: patch size 33
    n: number of patches in one direction 24
    """
    # Calculate s and m
    # Calculate s in terms of n and p and h
    s = (h - p + 1) // n
    # n = (h - p + 1) // s
    m = n * n

    # Create a grid of offsets
    y_offsets = torch.arange(0, n * s, s).repeat_interleave(n)
    x_offsets = torch.arange(0, n * s, s).repeat(n)

    # Create the binary tensor
    binary_tensor = torch.zeros(m, h, h)

    # Create indices for all patches at once
    for i in range(m):
        y_start = y_offsets[i]
        x_start = x_offsets[i]
        binary_tensor[i, y_start:y_start+p, x_start:x_start+p] = 1

    return binary_tensor