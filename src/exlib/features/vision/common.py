from collections import namedtuple
import torch.nn as nn
import torch
import numpy as np
from skimage.transform import resize
from skimage.measure import regionprops
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

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


def defragment_segments(segments: torch.LongTensor):
    """
    Sometimes the segment is not labeled in order (e.g., from SAM).
    Input:
        [[0, 1, 2],
         [4, 5, 6],
         [8, 9, 10]]

    Output:
        [[0, 1, 2],
         [3, 4, 5],
         [6, 7, 8]]
    """
    assert segments.ndim == 2
    uniques = segments.unique()
    new_segments = segments.clone()
    for i, k in enumerate(uniques):
        new_segments[segments == k] = i
    return new_segments


def relabel_segments_by_proximity(segments: torch.LongTensor):
    """
    Recall that many skimage segmentation methods return an integer-valued (H,W) array.
    However, these are often such that adjacent segments have very different integer labels.
    This is undesirable if we want a simple way of merging segments,
    e.g. reducing 32 segs to 16 by merging consecutive numbers.
    This function relabels the segments.
    
    Input:
        [[0, 8, 11,
         [7, 2, 6],
         [3, 5, 4]]

    Output (made-up example)
        [[0, 1, 8],
         [2, 3, 7],
         [4, 5, 6]]
    """
    assert segments.ndim == 2 # (H,W)
    device = segments.device
    segments = segments.cpu().numpy()
    props = regionprops(segments)
    centroids = np.array([prop.centroid for prop in props])
    
    # Pair-wise distance between centroids
    distance_matrix = squareform(pdist(centroids))
    
    # Hierarchical clustering
    Z = linkage(distance_matrix, method="single")
    
    # Get new labels that preserve proximity order
    new_labels = fcluster(Z, t=len(props), criterion="maxclust")

    # Re-map
    old_to_new = {prop.label: new_label for prop, new_label in zip(props, new_labels)}

    new_segments = np.copy(segments)
    for old_label, new_label in old_to_new.items():
        new_segments[segments == old_label] = new_label

    return torch.tensor(new_segments).to(device)

