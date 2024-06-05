import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy
import skimage

sys.path.append("../src")
import exlib
from exlib.datasets.chestx import ChestXDataset, ChestXPathologyModel, ChestXMetric

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


class GridGroups(nn.Module):
    # Let's assume image is 224x224 and make 28-wide grids (i.e., 8x8 partitions)
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, _, _, _ = x.shape
        mask_small = torch.tensor(range(8*8)).view(1,1,8,8).repeat(N,1,1,1)
        mask_big = F.interpolate(mask_small.float(), scale_factor=28).round().long()
        return mask_big.view(N,224,224)


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
        C, _, _ = image.shape
        if C == 1:
            image = image.repeat(3,1,1)
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
        

def get_chestx_scores(baselines = ['patch', 'quickshift', 'watershed']):
    dataset = ChestXDataset(split="test")
    pathols_model = ChestXPathologyModel().from_pretrained("BrachioLab/chestx_pathols").eval()

    metric = ChestXMetric()
    torch.manual_seed(1234)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    all_baselines_scores = {}
    for baseline in baselines:
        print('BASELINE:', baseline)
        if baseline == 'patch': # gridding 
            groups = GridGroups()
        elif baseline == 'quickshift': # quickshift
            groups = QuickShiftGroups()
        elif baseline == 'watershed': # watershift
            groups = WatershedGroups()
        
        scores = []
        for i, item in enumerate(tqdm(dataloader)):
            image = item["image"]
            with torch.no_grad():
                structs_masks = item["structs"]
                masks = F.one_hot(groups(image)).permute(0,3,1,2)
                score = metric(masks, structs_masks) # (N,H,W)
                scores.append(score.mean(dim=(1,2)))
            if i > 10:
                break 
        scores = torch.cat(scores)
        print(f"Avg alignment of {baseline} features: {scores.mean():.4f}")
        all_baselines_scores[baseline] = scores

    return all_baselines_scores
        

# if __name__ == "__main__": 
#     get_chestx_scores()
    