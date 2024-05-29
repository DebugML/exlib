import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels
import torchvision.transforms as transforms

import numpy as np

import glob
from typing import Optional, Union, List

import datasets as hfds
import huggingface_hub as hfhub

HF_DATA_REPO = "BrachioLab/cholecystectomy_segmentation"

class CholecDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_repo: str = HF_DATA_REPO,
        image_size: tuple[int] = (360, 640)
    ):
        self.dataset_dict = hfds.load_dataset(
            hf_repo,
            download_mode = hfds.DownloadMode.REUSE_CACHE_IF_EXISTS
        )
        self.dataset = self.dataset_dict["all_data"]
        self.image_size = image_size
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transforms(self.dataset[idx]["image"])
        gonogo = self.transforms(np.array(self.dataset[idx]["gonogo"]))
        organs = self.transforms(np.array(self.dataset[idx]["organs"]))
        return {
            "image": image,
            "gonogo": gonogo.view(image.shape[1:]),
            "organs": organs.view(image.shape[1:])
        }


class CholecModelOutput:
    def __init__(self, logits: torch.FloatTensor):
        self.logits = logits


class CholecModel(nn.Module, hfhub.PyTorchModelHubMixin):
    def __init__(self, task: str = "gonogo"):
        super().__init__()
        if task == "gonogo":
            self.num_labels = 3
        elif task == "organs":
            self.num_labels = 4
        else:
            raise ValueError(f"Unrecognized task {task}")

        self.seg_model = tvmodels.segmentation.fcn_resnet50(num_classes=self.num_labels)

    def forward(self, x: torch.FloatTensor):
        seg_out = self.seg_model(x)
        return CholecModelOutput(
            logits = seg_out["out"]
        )


class CholecAlignment(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        groups_pred: torch.LongTensor(),
        groups_true: torch.LongTensor()
    ):
        """
            groups_pred: (N,P,H W)
            groups_true: (N,T,H,W)
        """
        N, P, H, W = groups_pred.shape
        _, T, H, W = groups_true.shape

        # Make sure to binarize groups and shorten names to help with math
        Gp = groups_pred.bool().long()
        Gt = groups_true.bool().long()

        # Make (N,P,T)-shaped lookup tables for the intersection and union
        inters = (Gp.view(N,P,1,H,W) * Gt.view(N,1,T,H,W)).sum(dim=(-1,-2))
        unions = (Gp.view(N,P,1,H,W) + Gt.view(N,1,T,H,W)).clamp(0,1).sum(dim=(-1,-2))
        ious = inters / unions  # (N,P,T)
        ious[~ious.isfinite()] = 0 # Set the bad values to a score of zero
        iou_maxs = ious.max(dim=-1).values   # (N,P): max_{gt in Gt} iou(gp, gt)

        # sum_{gp in group_preds(feature)} iou_max(gp, Gt)
        pred_aligns_sum = (Gp * iou_maxs.view(N,P,1,1)).sum(dim=1) # (N,H,W)
        score = pred_aligns_sum / Gp.sum(dim=1) # (N,H,W), division is the |Gp(feaure)|
        score[~score.isfinite()] = 0    # Make div-by-zero things zero
        return score


