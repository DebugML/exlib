from collections import namedtuple
import torch.nn as nn
import torch
from skimage.transform import resize

SegmenterOutput = namedtuple("SegmenterOutput", ["segments", "segmenter_output"])

class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()
    

def compress_masks(data, resize_to=None, min_size=0):
    if resize_to is not None:
        data_resized = torch.stack([(torch.tensor(resize(mask, 
                                                        (resize_to, resize_to), 
                                                        preserve_range=True)) > 0.5).int() \
                            for mask in data.cpu().numpy()])
        data = data_resized

    data_count = data.sum(dim=-1).sum(dim=-1)
    _, sorted_indices = torch.sort(data_count, descending=True)

    data_sorted = data[sorted_indices]  # sorted masks

    masks = torch.zeros(data.shape[-2], data.shape[-1])

    count = 1
    for mask in data_sorted:
        new_mask = torch.logical_or(mask.bool(), torch.logical_and(mask.bool(), masks.bool()))
        if torch.sum(new_mask) >= min_size:
            masks[new_mask] = count
            count += 1

    masks = masks - 1
    masks = masks.int()
    masks[masks == -1] = torch.max(masks) + 1
    
    return masks