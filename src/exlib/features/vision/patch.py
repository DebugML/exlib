import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGroups(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        N, C, H, W = x.shape
        grid_height = (H // self.patch_size) + (H % self.patch_size)
        grid_width = (W // self.patch_size) + (W % self.patch_size)
        num_patches = grid_height * grid_width
        mask_small = torch.tensor(range(num_patches)).view(1,1,grid_height,grid_width).repeat(N,1,1,1)
        mask_big = F.interpolate(mask_small.float(), scale_factor=self.patch_size).round().long()
        return mask_big[:,:,:H,:W].view(N,H,W)

