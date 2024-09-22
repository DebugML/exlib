import torch
import torch.nn as nn 
    
class IdentityGroups(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return [torch.ones(len(x))]