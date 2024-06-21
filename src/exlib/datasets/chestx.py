import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tfs
import numpy as np
from dataclasses import dataclass
import torchxrayvision as xrv
import datasets as hfds
import huggingface_hub as hfhub

import sys
from tqdm import tqdm
sys.path.append("../src")
import exlib
from exlib.features.vision.patch import PatchGroups
from exlib.features.vision.quickshift import QuickshiftGroups
from exlib.features.vision.watershed import WatershedGroups


HF_DATA_REPO = "BrachioLab/chestx"

class ChestXDataset(torch.utils.data.Dataset):
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
class ChestXModelOutput:
    logits: torch.FloatTensor


class ChestXPathologyModel(nn.Module, hfhub.PyTorchModelHubMixin):
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


class ChestXMetric(nn.Module):
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

        # Make (N,P,T)-shaped lookup tables for the intersection-over-true
        inters = (Gp.view(N,P,1,H,W) * Gt.view(N,1,T,H,W)).sum(dim=(-1,-2))
        unions = (Gp.view(N,P,1,H,W) + Gt.view(N,1,T,H,W)).clamp(0,1).sum(dim=(-1,-2))
        iogs = inters / Gt.view(N,1,T,H,W).sum(dim=(-1,-2)) # (N,P,T)
        iogs[~iogs.isfinite()] = 0 # Set the bad values to a score of zero
        iog_maxs = iogs.max(dim=-1).values   # (N,P): max_{gt in Gt} iog(gp, gt)

        # sum_{gp in group_preds(feature)} iog_max(gp, Gt)
        pred_aligns_sum = (Gp * iog_maxs.view(N,P,1,1)).sum(dim=1) # (N,H,W)
        score = pred_aligns_sum / Gp.sum(dim=1) # (N,H,W), division is the |Gp(feaure)|
        score[~score.isfinite()] = 0    # Make div-by-zero things zero
        return score    # (N,H,W), a score for each feature



def get_chestx_scores(
    baselines = ['patch', 'quickshift', 'watershed'],
    dataset = ChestXDataset(split="test"),
    metric = ChestXMetric(),
    N = 100,
    batch_size = 4,
):
    dataset, _ = torch.utils.data.random_split(dataset, [N, len(dataset)-N])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    all_baselines_scores = {}
    for item in tqdm(dataloader):
        for baseline in baselines:
            if baseline == 'patch': # patch
                groups = PatchGroups()
            elif baseline == 'quickshift': # quickshift
                groups = QuickshiftGroups()
            elif baseline == 'watershed': # watershed
                groups = WatershedGroups()

            image = item["image"]
            with torch.no_grad():
                structs_masks = item["structs"]
                pred_masks = groups(image)
                score = metric(pred_masks, structs_masks) # (N,H,W)

                if baseline in all_baselines_scores.keys():
                    scores = all_baselines_scores[baseline]
                    scores.append(score.mean(dim=(1,2)))
                else: 
                    scores = [score.mean(dim=(1,2))]
                all_baselines_scores[baseline] = scores

    for baseline in baselines:
        scores = torch.cat(all_baselines_scores[baseline])
        # print(f"Avg alignment of {baseline} features: {scores.mean():.4f}")
        all_baselines_scores[baseline] = scores

    return all_baselines_scores
        


