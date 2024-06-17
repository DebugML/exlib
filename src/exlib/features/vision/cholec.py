import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage
import numpy as np

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

sys.path.append("../../src")
from exlib.datasets.cholec import CholecDataset, CholecModel, CholecMetric

from .patch import PatchGroups
from .quickshift import QuickshiftGroups
from .watershed import Watershedgroups


def get_cholec_scores(
    dataset = CholecDataset(split="test"),
    metric = CholecMetric(),
    baselines = ['patch', 'quickshift', 'watershed'],
    N = 100,
    batch_size = 4,
):
    dataset, _ = torch.utils.data.random_split(dataset, [N, len(dataset)-N])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    all_baselines_scores = {}
    for item in tqdm(dataloader):
        for baseline in baselines:
            if baseline == 'patch': # gridding 
                groups = PatchGroups()
            elif baseline == 'quickshift': # quickshift
                groups = QuickShiftGroups()
            elif baseline == 'watershed': # watershed
                groups = WatershedGroups()
    
            image = item["image"]
            with torch.no_grad():
                organs_masks = F.one_hot(item["organs"]).permute(0,3,1,2)
                masks = F.one_hot(groups(image)).permute(0,3,1,2)
                score = metric(masks, organs_masks) # (N,H,W)

                if baseline in all_baselines_scores.keys():
                    scores = all_baselines_scores[baseline]
                    scores.append(score.mean(dim=(1,2)))
                else: 
                    scores = [score.mean(dim=(1,2))]
                all_baselines_scores[baseline] = scores

    
    for baseline in baselines:
        scores = torch.cat(all_baselines_scores[baseline])
        print(f"Avg alignment of {baseline} features: {scores.mean():.4f}")
        all_baselines_scores[baseline] = scores

    return all_baselines_scores
    
    
