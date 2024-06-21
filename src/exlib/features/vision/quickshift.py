import torch
import torch.nn as nn
import numpy as np
from skimage.segmentation import quickshift
import torch.nn.functional as F


class QuickshiftGroups(nn.Module):
    # Use quickshift to perform image segmentation
    def __init__(self, kernel_size=10, max_dist=20, sigma=5, max_segs=64, flat=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.sigma = sigma
        self.max_segs = max_segs
        self.flat = flat

    def quickshift(self, image):
        # image is (C,H,W)
        assert image.ndim == 3

        # We have to make the images 3-channel for quickshift
        if image.size(0) == 1:
            image_np = image.repeat(3,1,1).numpy().transpose(1,2,0)
        elif image.size(0) == 3:
            image_np = image.numpy().transpose(1,2,0)
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")

        # quickshift returns a (H,W) of numpy integers
        segs = quickshift(
            image_np,
            kernel_size = self.kernel_size,
            max_dist = self.max_dist,
            sigma = self.sigma
        )

        segs = torch.tensor(segs)
        div_by = (segs.unique().max() / self.max_segs).long().item() + 1
        segs = segs // div_by
        return segs.long() # (H,W) of integers

    def forward(self, x):
        # x: (N,C,H,W)
        segs = torch.stack([self.quickshift(xi.cpu()) for xi in x]) # (N,H,W)
        if not self.flat:
            return F.one_hot(segs).permute(0,3,1,2).to(x.device) # (N,M,H,W)
        return segs.to(x.device)


