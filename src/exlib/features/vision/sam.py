import torch.nn as nn
import torch.nn.functional as F
import os
import requests
from typing import Optional
import shutil
import torch
import numpy as np
file_dir_path = os.path.dirname(os.path.realpath(__file__))

from segment_anything import sam_model_registry, SamPredictor
from segment_anything import build_sam, SamAutomaticMaskGenerator

from .common import defragment_segments, relabel_segments_by_proximity


DOWNLOAD_URLS = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
}

class SamGroups(nn.Module):
    def __init__(
        self,
        max_groups: int = 32,
        model_name: str = 'vit_h',
        model_dir: Optional[str] = None,
        download: bool = True,
        flat: bool = False,
    ):
        super().__init__()
        if model_dir is None:
            model_dir = os.path.join(file_dir_path, 'sam_models')
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        if download:
            self.download_model(model_name)

        url = DOWNLOAD_URLS[model_name]
        filename = os.path.join(self.model_dir, os.path.basename(url))
        self.sam = sam_model_registry[model_name](checkpoint=filename)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.max_groups = max_groups
        self.flat = flat

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

    @torch.no_grad()
    def forward(self, x):
        # x: (N,C,H,W)
        all_segs = []
        for xi in x:
            assert xi.ndim == 3
            if xi.size(0) == 1:
                xi = xi.repeat(3,1,1)
            xi_np = (xi * 255).byte().permute(1,2,0).cpu().numpy()
            outs = self.mask_generator.generate(xi_np)
            segs = sum([k * o["segmentation"] for (k,o) in enumerate(outs)])
            segs = torch.LongTensor(segs)
            segs = defragment_segments(segs) # SAM often skips labels
            segs = relabel_segments_by_proximity(segs)
            if segs.unique().max() + 1 >= self.max_groups:
                div_by = (segs.unique().max() + 1) / self.max_groups
                segs = segs // div_by
            all_segs.append(segs.long())

        all_segs = torch.stack(all_segs).to(x.device) # (N,H,W)
        if self.flat:
            return all_segs
        else:
            return F.one_hot(all_segs, num_classes=self.max_groups).permute(0,3,1,2) # (N,M,H,W)


