import torch
import torch.nn as nn 

class SentenceGroups(nn.Module):
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
        masks = []
        # reconstruct sentences from word list 
        # (different for each example)
        new_sentence = True
        for word_i in range(len(x)):
            if x[word_i] == '':
                break
            if new_sentence:
                new_sentence = False
                mask = torch.zeros(len(x)).long()
            mask[word_i] = 1
            if x[word_i][-1] in [".", "!", "?"]:
                masks.append(mask)
                new_sentence = True # so that the next word will be in the next sentence
        
        # pick n random sentence groups
        ngroups = min(self.max_groups, len(masks))
        r = torch.randperm(len(masks))[:ngroups]
        masks = [masks[pos] for pos in r]
        return masks
