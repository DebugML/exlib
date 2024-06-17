import torch
import torch.nn as nn
import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


class WatershedGroups(nn.Module):
    def __init__(self, fp_size=10, min_dist=20, compactness=10, max_segs=64):
        """
        compactness: Higher values result in more regularly-shaped watershed basins.
        """
        super().__init__()
        self.fp_size = fp_size
        self.min_dist = min_dist
        self.compactness = compactness
        self.max_segs = max_segs

    def watershed(self, image):
        # image is (C,H,W)
        image = (image.mean(dim=0).numpy() * 255).astype(np.uint8)
        distance = ndi.distance_transform_edt(image)
        coords = peak_local_max(
            distance,
            min_distance=self.min_dist,
            footprint=np.ones((self.fp_size,self.fp_size)),
            labels=image,
        )
        # coords = peak_local_max(distance, min_distance=10, labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        segs = watershed(
            -distance,
            markers,
            mask=image,
            compactness = self.compactness
        )
        segs = torch.tensor(segs)
        div_by = (segs.unique().max() / self.max_segs).long().item() + 1
        segs = segs // div_by
        return segs.long() # (H,W) of integers

    def forward(self, x):
        # x: (N,C,H,W)
        segs = torch.stack([self.watershed(xi.cpu()) for xi in x]) # (N,H,W)
        return segs.to(x.device)
