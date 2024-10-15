import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from exlib.explainers.archipelago import ArchipelagoTimeSeriesCls
from skimage.segmentation import quickshift
import numpy as np

class ArchipelagoGroups(nn.Module):
    def __init__(
        self,
        feature_extractor = torchvision.models.resnet18(pretrained=True).eval(),
        quickshift_kwargs = {
            "kernel_size": 8,
            "max_dist": 100.,
            "ratio": 0.2,
            "sigma": 10.
        },
        max_groups: int = 9,
        flat: bool = False, labels=None, past_values=None, past_time_features=None, past_observed_mask=None
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.quickshift_kwargs = quickshift_kwargs


        def segmenter(time_series, window_size = 50):
            num_segments = time_series.shape[-2] // window_size
            segments = np.zeros(time_series.shape[-2], dtype=np.int64)
            for i in range(num_segments):
                segments[i * window_size:(i + 1) * window_size] = i
            duplicated_segments = np.tile(segments, (time_series.shape[-3], 1))
            return duplicated_segments

        self.archipelago = ArchipelagoTimeSeriesCls(
            model=self.feature_extractor,
            top_k=max_groups,
            segmenter=segmenter
        )
        self.flat = flat

    def forward(self, labels=None, past_values=None, past_time_features=None, past_observed_mask=None):

        x = torch.cat([past_values.permute(0, 2, 1), past_time_features.permute(0, 2, 1), past_observed_mask], dim=1)
        x = x.unsqueeze(1)
        
        N, C, H, W = x.shape
        results = self.archipelago(x)
        segs = results.group_masks.squeeze(1)
        groups = F.one_hot(segs).permute(0,3,1,2).to(x.device) # (N,M,H,W)
        groups_reshaped = torch.flatten(groups, start_dim=1, end_dim=2)
        return groups_reshaped