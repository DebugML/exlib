import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGroups(nn.Module):
    def __init__(self, patch_size=16, num_patches=(14, 14), mode='size', flat=False):
        super().__init__()
        assert mode in ['size', 'count']
        self.patch_size = patch_size
        self.mode = mode
        self.flat = flat
        self.num_patches = num_patches

    def forward(self, x):
        N, C, H, W = x.shape
        if self.mode == 'size':
            grid_height = (H // self.patch_size) + (H % self.patch_size)
            grid_width = (W // self.patch_size) + (W % self.patch_size)
        else:
            grid_height = self.num_patches[0]
            grid_width = self.num_patches[1]
        num_patches = grid_height * grid_width
        mask_small = torch.tensor(range(num_patches)).view(1,1,grid_height,grid_width).repeat(N,1,1,1)
        if self.mode == 'size':
            mask_big = F.interpolate(mask_small.float(), scale_factor=self.patch_size).round().long()
        else:
            mask_big = F.interpolate(mask_small.float(), size=(H,W), mode='nearest').round().long()
        segs = mask_big[:,:,:H,:W].view(N,H,W)
        if not self.flat:
            return F.one_hot(segs).permute(0,3,1,2).to(x.device) # (N,M,H,W)
        return segs.to(x.device)

