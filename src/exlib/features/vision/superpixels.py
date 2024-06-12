import numpy as np
import torch
import torch.nn as nn

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from .common import *

class Superpixels(nn.Module):
    def __init__(self, method, **kwargs):
        super().__init__()
        self.method = method
        self.kwargs = kwargs
        self.set_defaults()

    def set_defaults(self):
        if self.method == 'felzenszwalb':
            self.kwargs.setdefault('scale', 100)
            self.kwargs.setdefault('sigma', 0.5)
            self.kwargs.setdefault('min_size', 50)
        elif self.method == 'slic':
            self.kwargs.setdefault('n_segments', 250)
            self.kwargs.setdefault('compactness', 10)
            self.kwargs.setdefault('sigma', 1)
            self.kwargs.setdefault('start_label', 1)
        elif self.method == 'quickshift':
            self.kwargs.setdefault('kernel_size', 3)
            self.kwargs.setdefault('max_dist', 6)
            self.kwargs.setdefault('ratio', 0.5)
        elif self.method == 'watershed':
            self.kwargs.setdefault('markers', 250)
            self.kwargs.setdefault('compactness', 0.001)
        else:
            raise ValueError(f'Unknown method: {self.method}')

    def forward(self, x):
        device = x.device
        bsz = x.shape[0]
        segments = torch.zeros(bsz, 1, *x.shape[-2:], dtype=int)
        for i in range(bsz):
            segments[i] = torch.from_numpy(self._segment(x[i].cpu().permute(1,2,0).numpy())).to(device)
        return SegmenterOutput(segments, {})

    def _segment(self, x):
        if self.method == 'felzenszwalb':
            return felzenszwalb(x, **self.kwargs)
        elif self.method == 'slic':
            return slic(x, **self.kwargs)
        elif self.method == 'quickshift':
            return quickshift(x, **self.kwargs)
        elif self.method == 'watershed':
            gradient = sobel(rgb2gray(x))
            return watershed(gradient, **self.kwargs)
        else:
            raise ValueError(f'Unknown method: {self.method}')


class Felzenszwalb(Superpixels):
    def __init__(self, **kwargs):
        super().__init__('felzenszwalb', **kwargs)


class Slic(Superpixels):
    def __init__(self, **kwargs):
        super().__init__('slic', **kwargs)


class Quickshift(Superpixels):
    def __init__(self, **kwargs):
        super().__init__('quickshift', **kwargs)


class Watershed(Superpixels):
    def __init__(self, **kwargs):
        super().__init__('watershed', **kwargs)

