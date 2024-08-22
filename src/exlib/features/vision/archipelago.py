import torch
import torch.nn as nn
import torchvision

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from explainers.archipelago import ArchipelagoImageCls

class ArchipelagoGroups(nn.Module):
    def __init__(
        self,
        feature_extractor = torchvision.models.resnet18(pretrained=True).eval(),
        quickshift_kwargs = {
            "kernel_size": 8,
            "max_dist": 100.,
            "ratio": 0.2,
            "sigma": 10.
        }
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        quickshift_kwargs = quickshift_kwargs

    def forward(self, x: torch.FloatTensor):
        pass



