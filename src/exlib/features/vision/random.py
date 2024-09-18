import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomGroups(nn.Module):
    def __init__(
        self,
        max_groups: int = 16,
        flat: bool = False
    ):
        super().__init__()
        self.max_groups = max_groups
        self.flat = flat

    def forward(self, x):
        N, C, H, W = x.shape
        segs = torch.randint(0, self.max_groups, (N,H,W)).to(x.device)
        if self.flat:
            return segs
        else:
            return F.one_hot(segs, num_classes=self.max_groups).permute(0,3,1,2) # (N,M,H,W)


