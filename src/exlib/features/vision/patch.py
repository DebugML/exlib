from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGroups(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, tuple[int,int]] = 16,
        flat: bool = False
    ):
        super().__init__()
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size
        self.flat = flat

    def forward(self, x):
        N, C, H, W = x.shape
        grid_height = (H // self.patch_size[0]) + (H % self.patch_size[0])
        grid_width = (W // self.patch_size[1]) + (W % self.patch_size[1])
        num_patches = grid_height * grid_width
        mask_small = torch.tensor(range(num_patches)).view(1,1,grid_height,grid_width).repeat(N,1,1,1)
        mask_big = F.interpolate(mask_small.float(), scale_factor=self.patch_size).round().long()
        segs = mask_big[:,:,:H,:W].view(N,H,W)
        if self.flat:
            return segs.to(x.device)
        else:
            return F.one_hot(segs).permute(0,3,1,2).to(x.device) # (N,M,H,W)

