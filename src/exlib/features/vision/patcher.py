import torch.nn as nn
import torch
import torch.nn.functional as F

from .common import *


class PatchSegmenter(nn.Module):
    def __init__(self, sz=(16, 16)):
        super().__init__()
        if isinstance(sz, int):
            sz = (sz, sz)
        self.sz = sz
    
    def forward(self, x):
        bsz = x.shape[0]
        segments_all = torch.zeros(bsz, 1, *x.shape[-2:], dtype=int)
        for i in range(bsz):
            idx = torch.arange(self.sz[0]*self.sz[1]).view(1,1,*self.sz).float()
            segments = F.interpolate(idx, size=x.size()[-2:], mode='nearest').long()
            segments_all[i] = segments[0]
        return SegmenterOutput(segments_all, {})