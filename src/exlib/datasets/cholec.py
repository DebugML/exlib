import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import torchvision.models as tvm
import torchvision.transforms as tfs
from dataclasses import dataclass
import datasets as hfds
import huggingface_hub as hfhub

import sys
from tqdm import tqdm
sys.path.append("../src")
import exlib
from exlib.features.vision import *

HF_DATA_REPO = "BrachioLab/cholecystectomy_segmentation"

class CholecDataset(torch.utils.data.Dataset):
    gonogo_names: str = [
        "Background",
        "Safe",
        "Unsafe"
    ]

    organ_names: str = [
        "Background",
        "Liver",
        "Gallbladder",
        "Hepatocystic Triangle"
    ]

    def __init__(
        self,
        split: str = "train",
        hf_data_repo: str = HF_DATA_REPO,
        image_size: tuple[int] = (360, 640)
    ):
        self.dataset = hfds.load_dataset(hf_data_repo, split=split)
        self.dataset.set_format("torch")
        self.image_size = image_size
        self.preprocess_image = tfs.Compose([
            tfs.Lambda(lambda x: x.float() / 255),
            tfs.Resize(image_size),
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
        gonogo = self.preprocess_labels(self.dataset[idx]["gonogo"])
        organs = self.preprocess_labels(self.dataset[idx]["organs"])
        return {
            "image": image,     # (3,H,W)
            "gonogo": gonogo,   # (H,W)
            "organs": organs,   # (H,W)
        }


@dataclass
class CholecModelOutput:
    logits: torch.FloatTensor


class CholecModel(nn.Module, hfhub.PyTorchModelHubMixin):
    def __init__(self, task: str = "gonogo"):
        super().__init__()
        self.task = task
        if task == "gonogo":
            self.num_labels = 3
        elif task == "organs":
            self.num_labels = 4
        else:
            raise ValueError(f"Unrecognized task {task}")

        self.seg_model = tvm.segmentation.fcn_resnet50(num_classes=self.num_labels)
        self.preprocess = tfs.Compose([
            tfs.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def forward(self, x: torch.FloatTensor):
        x = self.preprocess(x)
        seg_out = self.seg_model(x)
        return CholecModelOutput(
            logits = seg_out["out"]
        )


class CholecFixScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        groups_pred: torch.LongTensor,
        groups_true: torch.LongTensor,
        big_batch: bool = False,
        reduce: bool = True
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
        if big_batch:
            inters = (Gp.view(N,P,1,H,W) * Gt.view(N,1,T,H,W)).sum(dim=(-1,-2))
            unions = (Gp.view(N,P,1,H,W) + Gt.view(N,1,T,H,W)).clamp(0,1).sum(dim=(-1,-2))
        else:
            # More memory-efficient
            inters = torch.zeros(N,P,T).to(Gp.device)
            unions = torch.zeros(N,P,T).to(Gp.device)
            for i in range(P):
                for j in range(T):
                    inters[:,i,j] = (Gp[:,i] * Gt[:,j]).sum(dim=(-1,-2))
                    unions[:,i,j] = (Gp[:,i] + Gt[:,j]).clamp(0,1).sum(dim=(-1,-2))
        ious = inters / unions  # (N,P,T)
        ious[~ious.isfinite()] = 0 # Set the bad values to a score of zero
        iou_maxs = ious.max(dim=-1).values   # (N,P): max_{gt in Gt} iou(gp, gt)

        # sum_{gp in group_preds(feature)} iou_max(gp, Gt)
        pred_aligns_sum = (Gp * iou_maxs.view(N,P,1,1)).sum(dim=1) # (N,H,W)
        score = pred_aligns_sum / Gp.sum(dim=1) # (N,H,W), division is the |Gp(feaure)|
        score[~score.isfinite()] = 0    # Make div-by-zero things zero
        if reduce:
            return score.mean(dim=(1,2))
        else:
            return score    # (N,H,W), a score for each feature


def get_cholec_scores(
    baselines = ['identity', 'random', 'patch', 'quickshift', 'watershed', 'sam', 'ace', 'craft', 'archipelago'],
    N = None,
    batch_size = 8,
    device = "cuda" if torch.cuda.is_available() else "cpu",
):
    torch.manual_seed(1234)
    dataset = CholecDataset(split="test")
    metric = CholecFixScore()

    if N is not None:
        dataset = Subset(dataset, torch.randperm(len(dataset))[:N].tolist())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    resizer = tfs.Resize((180,320)) # Originally (360,640)
    
    all_baselines_scores = {}
    for item in tqdm(dataloader):
        for baseline in baselines:
            if baseline == "identity":
                groups = IdentityGroups()
            elif baseline == "random":
                groups = RandomGroups(max_groups=8)
            elif baseline == "patch": # patch
                groups = PatchGroups(grid_size=(8,14), mode="grid")
            elif baseline == "quickshift": # quickshift
                groups = QuickshiftGroups(max_groups=8)
            elif baseline == "watershed": # watershed
                groups = WatershedGroups(max_groups=8)
            elif baseline == "sam": # watershed
                groups = SamGroups(max_groups=8)
            elif baseline == "ace":
                groups = NeuralQuickshiftGroups(max_groups=8)
            elif baseline == "craft":
                groups = CraftGroups(max_groups=8)
            elif baseline == "archipelago":
                groups = ArchipelagoGroups(max_groups=8)

            groups.eval().to(device)

            image = resizer(item["image"].to(device))

            with torch.no_grad():
                organ_masks = F.one_hot(item["organs"]).permute(0,3,1,2).to(device)
                organ_masks = resizer(organ_masks.float()).long()
                pred_masks = groups(image)
                score = metric(pred_masks, organ_masks).cpu() # (N,H,W)

                if baseline in all_baselines_scores.keys():
                    scores = all_baselines_scores[baseline]
                    scores.append(score) #.mean(dim=(1,2)))
                else: 
                    scores = [score] #.mean(dim=(1,2))]
                all_baselines_scores[baseline] = scores

    
    for baseline in baselines:
        scores = torch.cat(all_baselines_scores[baseline])
        # print(f"Avg alignment of {baseline} features: {scores.mean():.4f}")
        all_baselines_scores[baseline] = scores

    return all_baselines_scores
    
    
