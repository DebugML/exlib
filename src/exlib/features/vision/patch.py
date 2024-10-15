import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGroups(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int,int] = (32,32),
        grid_size: int | tuple[int,int] = (8,8),
        flat: bool = False,
        mode: str = "grid"
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.mode = mode
        self.flat = flat

    def forward(self, x):
        N, C, H, W = x.shape
        if self.mode == "grid":
            gH, gW = self.grid_size
            pH = (H // gH) + (H % gH != 0)
            pW = (W // gW) + (W % gW != 0)
        elif self.mode == "patch":
            pH, pW = self.patch_size
            gH = (H // pH) + (H % pH != 0)
            gW = (W // pW) + (W % pW != 0)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        num_patches = gH * gW
        mask_small = torch.tensor(range(num_patches)).view(1,1,gH,gW).repeat(N,1,1,1)
        mask_big = F.interpolate(mask_small.float(), scale_factor=(pH,pW)).round().long()
        segs = mask_big[:,:,:H,:W].view(N,H,W)

        if self.flat:
            return segs.to(x.device)
        else:
            return F.one_hot(segs).permute(0,3,1,2).to(x.device) # (N,M,H,W)

