import sys
sys.path.append('../src')
import exlib
import torch

# Baseline
from skimage.segmentation import watershed, quickshift
from scipy import ndimage
from skimage.feature import peak_local_max
from exlib.explainers.common import convert_idx_masks_to_bool, patch_segmenter
import torch
import torch.nn as nn
import numpy as np
import cv2


class MassMapsPatch(nn.Module):
    def __init__(self, sz=(8, 8)):
        """
        sz : int, number of patches per side.
        """
        super().__init__()
        self.sz = sz
    
    def apply_patch(self, image):
        return patch_segmenter(image, sz=self.sz)
    
    def forward(self, images):
        """
        input: images (N, C=1, H, W)
        output: daf_preds (N, M, H, W)
        """
        daf_preds = []
        for image in images:
            segment_mask = torch.tensor(self.apply_patch(image[0].cpu().numpy())).to(images.device)
            masks_bool = convert_idx_masks_to_bool(segment_mask[None])
            daf_preds.append(masks_bool)
        daf_preds = torch.nn.utils.rnn.pad_sequence(daf_preds, batch_first=True)
        return daf_preds
        

class MassMapsQuickshift(nn.Module):
    def __init__(self, ratio=1.0, kernel_size=5, max_dist=10):
        """
        ratio : float, optional, between 0 and 1
            Balances color-space proximity and image-space proximity.
            Higher values give more weight to color-space.
        kernel_size : float, optional
            Width of Gaussian kernel used in smoothing the
            sample density. Higher means fewer clusters.
        max_dist : float, optional
            Cut-off point for data distances.
            Higher means fewer clusters.
        """
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        
    def apply_quickshift(self, image):
        ratio = self.ratio
        kernel_size = self.kernel_size
        max_dist = self.max_dist
        
        image = (image * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        raw_labels = quickshift(image_bgr, ratio=ratio, 
                                kernel_size=kernel_size, 
                                max_dist=max_dist)
        return raw_labels
    
    def forward(self, images):
        """
        input: images (N, C=1, H, W)
        output: daf_preds (N, M, H, W)
        """
        daf_preds = []
        for image in images:
            segment_mask = torch.tensor(self.apply_quickshift(image[0].cpu().numpy())).to(images.device)
            masks_bool = convert_idx_masks_to_bool(segment_mask[None])
            daf_preds.append(masks_bool)
        daf_preds = torch.nn.utils.rnn.pad_sequence(daf_preds, batch_first=True)
        return daf_preds


class MassMapsWatershed(nn.Module):
    def __init__(self, compactness=0, normalize=False):
        """
        compactness: Higher values result in more regularly-shaped watershed basins.
        """
        super().__init__()
        self.compactness = compactness
        self.normalize = normalize
        
    def apply_watershed(self, image):
        compactness = self.compactness
        normalize = self.normalize
        
        if normalize:
            # print('min', image.min(), 'max', image.max())
            image = (image - image.min()) / (image.max() - image.min())
            # print('after: min', image.min(), 'max', image.max())
        
        image = (image * 255).astype(np.uint8)
        distance = ndimage.distance_transform_edt(image)
        coords = peak_local_max(distance, min_distance=10, labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        raw_labels = watershed(-distance, markers, mask=image,
                               compactness=compactness)
        return raw_labels
    
    def forward(self, images):
        """
        input: images (N, C=1, H, W)
        output: daf_preds (N, M, H, W)
        """
        daf_preds = []
        for image in images:
            segment_mask = torch.tensor(self.apply_watershed(image[0].cpu().numpy())).to(images.device)
            masks_bool = convert_idx_masks_to_bool(segment_mask[None])
            daf_preds.append(masks_bool)
        daf_preds = torch.nn.utils.rnn.pad_sequence(daf_preds, batch_first=True)
        return daf_preds


class MassMapsOracle(nn.Module):
    def __init__(self, sz=(8, 8)):
        """
        sz : int, number of patches per side.
        """
        super().__init__()
        self.sz = sz
    
    def forward(self, images):
        """
        input: images (N, C=1, H, W)
        output: daf_preds (N, M, H, W)
        """
        daf_preds = []
        stds = torch.std(images.flatten(2), dim=-1)
        N, C, H, W = images.shape
        # import pdb; pdb.set_trace()
        masks = torch.zeros(N, 3, H, W).to(images.device)
        masks[:,0] = (images < 0)[:,0]
        masks[:,1] = (images > 3 * stds[:,:,None,None])[:,0]
        masks[:,2] = torch.logical_not(masks[:,1].logical_or(masks[:,0]))
        daf_preds = masks
        return daf_preds


class MassMapsOne(nn.Module):
    def __init__(self, sz=(8, 8)):
        """
        sz : int, number of patches per side.
        """
        super().__init__()
        self.sz = sz
    
    def forward(self, images):
        """
        input: images (N, C=1, H, W)
        output: daf_preds (N, M, H, W)
        """
        daf_preds = []
        stds = torch.std(images.flatten(2), dim=-1)
        N, C, H, W = images.shape
        masks = torch.ones(N, 1, H, W).to(images.device)
        daf_preds = masks
        return daf_preds

