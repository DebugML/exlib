import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tfs
from dataclasses import dataclass
import datasets as hfds
import huggingface_hub as hfhub

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
        split: str = "all_data",
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

        self.seg_model = torchvision.models.segmentation.fcn_resnet50(num_classes=self.num_labels)
        self.preprocess = tfs.Compose([
            tfs.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def forward(self, x: torch.FloatTensor):
        x = self.preprocess(x)
        seg_out = self.seg_model(x)
        return CholecModelOutput(
            logits = seg_out["out"]
        )


class CholecMetric(nn.Module):
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


