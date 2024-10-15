import torch
import torch.nn as nn
import torchvision
import timm

import sys
sys.path.append("../../..")
from exlib.explainers.libs.craft.craft_torch import Craft

class CraftGroups(nn.Module):
    def __init__(
        self,
        max_groups: int = 16,
        number_of_concepts: int = 10,
        batch_size: int = 4,
    ):
        super().__init__()
        self.model = timm.create_model("nf_resnet50.ra2_in1k", pretrained=True).eval()
        self.g = nn.Sequential(*(list(self.model.children())[:4]))  # input to penultimate layer
        self.h = lambda x: self.model.head.fc(x.mean(dim=(2,3)))    # penultimate to logits
        self.max_groups = max_groups
        self.number_of_concepts = number_of_concepts
        self.batch_size = batch_size

    @torch.no_grad()
    def forward(self, x: torch.FloatTensor):
        N, C, H, W = x.shape
        
        if C == 1:
            x = x.repeat(1,3,1,1)

        d = min(H, W)
        patch_size = d // 5

        craft = Craft(
            input_to_latent = self.g,
            latent_to_logit = self.h,
            number_of_concepts = self.number_of_concepts,
            patch_size = int(patch_size),
            batch_size = self.batch_size,
            device = x.device
        )

        shrink = torchvision.transforms.Resize((d,d))
        x_shrunk = shrink(x)

        _, U, _, masks = craft.fit(x_shrunk, return_patch_masks=True)
        U = U.to(x.device)
        masks = masks.to(x.device).long()

        # Can directly take max because CRAFT uses a non-negative factorization
        topks = U.max(dim=-1).values.argsort(dim=-1)[:,:self.max_groups]    # (N,max_groups)
        selected_masks = masks[topks]   # (N,max_groups,d,d)
        grow = torchvision.transforms.Resize((H,W))
        return grow(selected_masks) # (N,max_groups,H,W)


