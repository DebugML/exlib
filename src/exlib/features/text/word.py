import torch
import torch.nn as nn

class WordGroups(nn.Module):
    def __init__(
        self, 
        distinct: int,
        scaling = 1.5
    ):
        super().__init__()
        self.scaling = scaling
        self.distinct = distinct
        self.max_groups = int(scaling * distinct)

    def forward(self, x): # x is word_list
        ngroups = min(self.max_groups, len(x))
        r = torch.randperm(len(x))[:ngroups]
        masks = []
        for pos in r:
            mask = torch.zeros(len(x)).long()
            mask[pos] = 1
            masks.append(mask)
        return masks
