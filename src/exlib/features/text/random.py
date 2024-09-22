import torch
import torch.nn as nn
import random

class RandomGroups(nn.Module):
    def __init__(
        self, 
        distinct: int,
        scaling = 1.5
    ):
        super().__init__()
        self.scaling = scaling
        self.distinct = distinct
        self.max_groups = int(scaling * distinct)

    def forward(self, x):
        ngroups = min(random.randint(1, self.max_groups), len(x))
        r = torch.randperm(len(x))
        all_groups_pos = torch.tensor_split(r, ngroups)
        masks = []
        for group_pos in all_groups_pos:
            mask = torch.zeros(len(x))
            for pos in group_pos:
                mask[pos] = 1
            masks.append(mask)
        return masks