import torch.nn as nn
import os
import requests
import shutil
import torch
import numpy as np
file_dir_path = os.path.dirname(os.path.realpath(__file__))

from segment_anything import sam_model_registry, SamPredictor
from segment_anything import build_sam, SamAutomaticMaskGenerator

from .common import *


DOWNLOAD_URLS = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
}

class SamSegmenter(nn.Module):
    def __init__(self, model_name='vit_h', points_per_side=32, model_dir=None, seg_min_size=100, download=True):
        super().__init__()
        if model_dir is None:
            model_dir = os.path.join(file_dir_path, 'sam_models')
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.seg_min_size = seg_min_size
        if download:
            self.download_model(model_name)

        url = DOWNLOAD_URLS[model_name]
        filename = os.path.join(self.model_dir, os.path.basename(url))
        self.sam = sam_model_registry[model_name](checkpoint=filename)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, 
                                                        points_per_side=points_per_side)

    def download_model(self, model_name):
        url = DOWNLOAD_URLS[model_name]
        filename = os.path.join(self.model_dir, os.path.basename(url))
        if os.path.exists(filename):
            return
        print(f'Downloading model... {model_name}')
        with requests.get(url, stream=True) as r:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print('Done downloading model.')

    def forward(self, x):
        bsz = x.shape[0]
        masks_all = []
        segments_all = []
        for i in range(bsz):
            image = x[i]
            masks = self.mask_generator.generate((image * 255).byte().permute(1,2,0).cpu().numpy())
            if len(masks) == 0:
                masks_seg = torch.zeros(image.shape[0], image.shape[2], image.shape[3]).bool()
            else:
                masks_seg = np.array([mask['segmentation'] for mask in masks])
                masks_seg = torch.from_numpy(masks_seg).bool()
            
            masks_all.append(masks_seg)
            segments = compress_masks(masks_seg, min_size=self.seg_min_size)
            segments_all.append(segments)
        segments_all = torch.stack(segments_all, dim=0)

        return SegmenterOutput(segments_all, {'segments_all': masks_all})