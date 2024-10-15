import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from explainers.archipelago import ArchipelagoImageCls
from skimage.segmentation import quickshift
import numpy as np

class ArchipelagoGroups(nn.Module):
    def __init__(
        self,
        feature_extractor = torchvision.models.resnet18(pretrained=True).eval(),
        quickshift_kwargs = {
            "kernel_size": 8,
            "max_dist": 100.,
            "ratio": 0.2,
            "sigma": 10.
        },
        max_groups: int = 16,
        segmenter = None,
        flat: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.quickshift_kwargs = quickshift_kwargs

        if segmenter is None:
            def segmenter(image):
                # Convert PyTorch tensor to NumPy array if necessary
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()

                # Ensure the image is 3D (H, W, C)
                if image.ndim == 2:
                    image = image[:, :, np.newaxis]
                elif image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W) format
                    image = np.transpose(image, (1, 2, 0))

                # Broadcast to 3 channels if necessary
                if image.shape[2] == 1:
                    image = np.broadcast_to(image, (*image.shape[:2], 3))

                # Ensure the image is in the correct dtype
                image = (image * 255).astype(np.uint8)

                seg_results = quickshift(image, **quickshift_kwargs)

                # print('seg_results', torch.tensor(seg_results).unique().shape)

                return seg_results

        self.archipelago = ArchipelagoImageCls(
            model=self.feature_extractor,
            top_k=max_groups,
            # segmenter='patch'
            segmenter=segmenter
        )
        self.max_groups = max_groups
        self.flat = flat

    def forward(self, x: torch.FloatTensor):
        N, C, H, W = x.shape    # Assume 4-dimensional

        results = self.archipelago(x)
        segs = results.group_masks.squeeze(1)

        if segs.unique().max() + 1 >= self.max_groups:
            div_by = (segs.unique().max() + 1) / self.max_groups
            segs = segs // div_by
            segs = segs.long()

        # import pdb; pdb.set_trace()
        if self.flat:
            # segs[segs >= self.max_groups] = -1
            return segs.to(x.device) # (N,H,W)
        else:
            return F.one_hot(segs).permute(0,3,1,2).to(x.device) #[:,:self.max_groups] # (N,M,H,W)


