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

sys.path.append("../../src")
import exlib
from exlib.datasets.chestx import ChestXDataset, ChestXPathologyModel, ChestXMetric

from .patch import PatchGroups
from .quickshift import QuickshiftGroups
from .watershed import WatershedGroups


def get_chestx_scores(
    dataset = ChestXDataset(split="test"),
    metric = ChestXMetric(),
    baselines = ['patch', 'quickshift', 'watershed'],
    N = 100,
    batch_size = 4,
):
    dataset, _ = torch.utils.data.random_split(dataset, [N, len(dataset)-N])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    all_baselines_scores = {}
    for item in tqdm(dataloader):
        for baseline in baselines:
            if baseline == 'patch': # gridding 
                groups = PatchGroups()
            elif baseline == 'quickshift': # quickshift
                groups = QuickshiftGroups()
            elif baseline == 'watershed': # watershed
                groups = WatershedGroups()
    
            image = item["image"]
            with torch.no_grad():
                structs_masks = item["structs"]
                masks = F.one_hot(groups(image)).permute(0,3,1,2)
                score = metric(masks, structs_masks) # (N,H,W)

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
        

