import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tfs
import numpy as np
from dataclasses import dataclass
import datasets as hfds
import huggingface_hub as hfhub

HF_DATA_REPO = "BrachioLab/chestxdet"

class ChestXDetDataset(torch.utils.data.Dataset):
    pathology_names = [
        "Consolidation",
        "Diffuse Nodule",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Fracture",
        "Mass",
        "Nodule",
        "Pleural Thickening",
        "Pneumothorax"
    ]

    structure_names: str = [
        "Left Clavicle",
        "Right Clavicle",
        "Left Scapula",
        "Right Scapula",
        "Left Lung",
        "Right Lung",
        "Left Hilus Pulmonis",
        "Right Hilus Pulmonis",
        "Heart",
        "Aorta",
        "Facies Diaphragmatica",
        "Mediastinum",
        "Weasand",
        "Spine"
    ]

    def __init__(
        self,
        split: str = "train",
        hf_data_repo: str = HF_DATA_REPO,
        image_size: int = 1024,
    ):
        self.dataset = hfds.load_dataset(hf_data_repo, split=split)
        self.dataset.set_format("torch")
        self.image_size = image_size
        self.preprocess_image = tfs.Compose([
            tfs.Lambda(lambda x: x.float().mean(dim=0, keepdim=True) / 255),
            tfs.Resize(image_size)

        ])

        self.preprocess_labels = tfs.Compose([
            tfs.Lambda(lambda x: x.unsqueeze(0)),
            tfs.Resize(image_size),
            tfs.Lambda(lambda x: x[0])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.preprocess_image(self.dataset[idx]["image"])
        P = self.preprocess_labels(self.dataset[idx]["pathols"])
        S = self.preprocess_labels(self.dataset[idx]["structs"])
        pathols = torch.stack([(P // (2**i)) % 2 == 1 for i in range(13)]).long()
        structs = torch.stack([(S // (2**i)) % 2 == 1 for i in range(14)]).long()

        return {
            "image": image,     # (1,H,W)
            "pathols": pathols, # (13,H,W)
            "structs": structs, # (14,H,W)
        }


@dataclass
class ChestXDetModelOutput:
    logits: torch.FloatTensor


class ChestXDetModel(nn.Module, hfhub.PyTorchModelHubMixin):
    def __init__(
        self,
        task: str = "pathols",
        scaled_image_size: int = 256,
    ):
        super().__init__()
        self.task = task
        if task == "pathols":
            self.num_labels = 13
        elif task == "structs":
            self.num_labels = 14
        else:
            raise ValueError(f"Unrecognized task {task}")

        self.seg_model = torchvision.models.segmentation.fcn_resnet50(num_classes=self.num_labels)
        self.scaled_image_size = scaled_image_size
        self.preprocess = tfs.Compose([
            tfs.Normalize(mean=[0.511], std=[0.257]),    # Computed from 1000 samples of training data
            tfs.Resize(scaled_image_size)
        ])


    def forward(self, x: torch.FloatTensor):
        N, _, H, W = x.shape
        x = self.preprocess(x).repeat(1,3,1,1) # (N,3,H,W)
        seg_out = self.seg_model(x)
        logits = seg_out["out"]
        logits = tfs.Resize((H,W))(logits)
        return ChestXDetModelOutput(
            logits = logits
        )


