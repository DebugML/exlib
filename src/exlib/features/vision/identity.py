import torch
import torch.nn as nn

class IdentityGroups(nn.Module):
    def __init__(
        self,
        flat: bool = False
    ):
        super().__init__()
        self.flat = flat

    def forward(self, x):
        N, C, H, W = x.shape
        if self.flat:
            return torch.zeros(N,H,W).long().to(x.device)
        else:
            return torch.ones(N,1,H,W).long().to(x.device)


