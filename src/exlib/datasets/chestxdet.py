import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tfs
import numpy as np
from dataclasses import dataclass
import torchxrayvision as xrv
import datasets as hfds
import huggingface_hub as hfhub

HF_DATA_REPO = "BrachioLab/chestxdet"

class ChestXDetDataset(torch.utils.data.Dataset):
    pathology_names = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
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
        image_size: int = 224,
    ):
        self.dataset = hfds.load_dataset(hf_data_repo, split=split)
        self.dataset.set_format("torch")
        self.image_size = image_size
        self.preprocess_image = tfs.Compose([
            tfs.Lambda(lambda x: x.float().mean(dim=0, keepdim=True) / 255),
            tfs.Resize(image_size)

        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.preprocess_image(self.dataset[idx]["image"])
        pathols = torch.tensor(self.dataset[idx]["pathols"])
        structs = torch.tensor(self.dataset[idx]["structs"])

        return {
            "image": image,     # (1,H,W)
            "pathols": pathols, # (14)
            "structs": structs, # (14,H,W)
        }


@dataclass
class ChestXDetModelOutput:
    logits: torch.FloatTensor


class ChestXDetPathologyModel(nn.Module, hfhub.PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.xrv_model = xrv.models.DenseNet(weights="densenet121-res224-nih") # NIH chest X-ray8

    def forward(self, x: torch.FloatTensor):
        """ x: (N,C,224,224) with values in [0,1], with either C=1 or C=2 channels """

        x = x * 2048 - 1024 # The xrv model requires some brazingo scaling
        out = self.xrv_model(x)

        """ The XRV model outputs 18 pathology labels in the following order:
            ['Atelectasis',
             'Consolidation',
             'Infiltration',
             'Pneumothorax',
             'Edema',
             'Emphysema',
             'Fibrosis',
             'Effusion',
             'Pneumonia',
             'Pleural_Thickening',
             'Cardiomegaly',
             'Nodule',
             'Mass',
             'Hernia',
             '',
             '',
             '',
             '']
        ... so we need to sort it to match our ordering
        """
        pathol_idxs = [0, 10, 1, 4, 7, 5, 6, 13, 2, 12, 11, 9, 8, 3]
        return out[:,pathol_idxs]


class ChestXDetMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        groups_pred: torch.LongTensor(),
        groups_true: torch.LongTensor(),
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
        return score    # (N,H,W), a score for each feature
