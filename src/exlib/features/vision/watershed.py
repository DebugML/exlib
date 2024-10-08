import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
from scipy import ndimage as ndi

from .common import relabel_segments_by_proximity

class WatershedGroups(nn.Module):
    def __init__(
        self,
        max_groups: int = 16, 
        footprint_size: int = 10,
        min_dist: int = 20,
        compactness: float = 10.,
        normalize: bool = False,
        flat: bool = False
    ):
        """
        compactness: Higher values result in more regularly-shaped watershed basins.
        """
        super().__init__()
        self.max_groups = max_groups
        self.footprint_size = footprint_size
        self.min_dist = min_dist
        self.compactness = compactness
        self.normalize = normalize
        self.flat = flat

    def watershed(self, image):
        # image is (C,H,W)
        if self.normalize:
            image = (image - image.min()) / (image.max() - image.min())
        image = (image.mean(dim=0).numpy() * 255).astype(np.uint8)
        distance = ndi.distance_transform_edt(image)
        coords = skimage.feature.peak_local_max(
            distance,
            min_distance=self.min_dist,
            footprint=np.ones((self.footprint_size,self.footprint_size)),
            labels=image,
        )
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        segs = skimage.segmentation.watershed(
            -distance,
            markers,
            mask = image,
            compactness = self.compactness
        )
        segs = torch.tensor(segs)
        segs = relabel_segments_by_proximity(segs)

        if segs.unique().max() + 1 >= self.max_groups:
            div_by = (segs.unique().max() + 1) / self.max_groups
            segs = segs // div_by
        return segs.long() # (H,W) of integers

    def forward(self, x):
        # x: (N,C,H,W)
        segs = torch.stack([self.watershed(xi.cpu()) for xi in x]).to(x.device) # (N,H,W)
        if self.flat:
            return segs
        else:
            return F.one_hot(segs, num_classes=self.max_groups).permute(0,3,1,2) # (N,M,H,W)

