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
    def __init__(
        self,
        task: str = "gonogo"
    ):
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


