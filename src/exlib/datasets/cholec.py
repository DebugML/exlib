import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tfs
from dataclasses import dataclass
import datasets as hfds
import huggingface_hub as hfhub

import sys
from tqdm import tqdm
sys.path.append("../src")
import exlib
from exlib.features.vision.patch import PatchGroups
from exlib.features.vision.quickshift import QuickshiftGroups
from exlib.features.vision.watershed import WatershedGroups

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

        # Make (N,P,T)-shaped lookup tables for the intersection-over-trues
        inters = (Gp.view(N,P,1,H,W) * Gt.view(N,1,T,H,W)).sum(dim=(-1,-2))
        iogs = inters / Gt.view(N,1,T,H,W).sum(dim=(-1,-2)) # (N,P,T)
        iogs[~iogs.isfinite()] = 0 # Set the bad values to a score of zero
        iog_maxs = iogs.max(dim=-1).values   # (N,P): max_{gt in Gt} iog(gp, gt)

        # sum_{gp in group_preds(feature)} iou_max(gp, Gt)
        pred_aligns_sum = (Gp * iog_maxs.view(N,P,1,1)).sum(dim=1) # (N,H,W)
        score = pred_aligns_sum / Gp.sum(dim=1) # (N,H,W), division is the |Gp(feaure)|
        score[~score.isfinite()] = 0    # Make div-by-zero things zero
        return score    # (N,H,W), a score for each feature




def get_cholec_scores(
    baselines = ['patch', 'quickshift', 'watershed'],
    dataset = CholecDataset(split="test"),
    metric = CholecMetric(),
    N = 100,
    batch_size = 4,
):
    dataset, _ = torch.utils.data.random_split(dataset, [N, len(dataset)-N])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
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
                organs_masks = F.one_hot(item["organs"]).permute(0,3,1,2)
                pred_masks = groups(image)
                score = metric(pred_masks, organs_masks) # (N,H,W)

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
    
    