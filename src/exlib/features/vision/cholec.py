import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


class GridGroups(nn.Module):
    # Let's assume image is 360x640 and make 40x40 grids (i.e., 9x16 partitions)
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, _, _, _ = x.shape
        mask_small = torch.tensor(range(9*16)).view(1,1,9,16).repeat(N,1,1,1)
        mask_big = F.interpolate(mask_small.float(), scale_factor=40).round().long()
        return mask_big.view(N,360,640)

class QuickShiftGroups(nn.Module):
    # Use quickshift to perform image segmentation
    def __init__(self, kernel_size=10, max_dist=20, sigma=5, max_segs=40):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.sigma = sigma
        self.max_segs = max_segs

    def quickshift(self, image):
        # image is (C,H,W)
        image_np = image.numpy().transpose(1,2,0)
        segs = skimage.segmentation.quickshift(image_np, kernel_size=self.kernel_size, max_dist=self.max_dist, sigma=self.sigma)
        segs = torch.tensor(segs)
        segs[segs >= self.max_segs] = self.max_segs - 1
        return segs.long() # (H,W) of integers

    def forward(self, x):
        # x: (N,C,H,W)
        segs = torch.stack([self.quickshift(xi.cpu()) for xi in x]) # (N,H,W)
        return segs.to(x.device)
        

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
        segs = skimage.segmentation.watershed(
            -distance,
            markers,
            mask=image,
            compactness = self.compactness
        )
        # segs = skimage.segmentation.watershed(image_np, kernel_size=self.kernel_size, max_dist=self.max_dist, sigma=self.sigma)
        segs = torch.tensor(segs)
        div_by = (segs.unique().max() / self.max_segs).long().item() + 1
        segs = segs // div_by
        return segs.long() # (H,W) of integers

    def forward(self, x):
        # x: (N,C,H,W)
        segs = torch.stack([self.watershed(xi.cpu()) for xi in x]) # (N,H,W)
        return segs.to(x.device)
        

