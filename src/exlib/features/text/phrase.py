import torch
import torch.nn as nn

class PhraseGroups(nn.Module):
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
        # make all possible phrase groups
        masks = []
        counter = 0
        for word_i in range(0, len(x), 3):
            mask = torch.zeros(len(x)).long()
            # each group is 3 consecutive words
            diff = len(x) - word_i
            if diff >= 3:
                mask[word_i:word_i+3] = 1
            else:
                mask[word_i:len(x[0])] = 1
            masks.append(mask)
            counter += 1

        # pick n random phrase groups
        ngroups = min(self.max_groups, len(masks))
        r = torch.randperm(len(masks))[:ngroups]
        masks = [masks[pos] for pos in r]
        return masks
    

