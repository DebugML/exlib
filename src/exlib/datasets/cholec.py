import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.models as tvm
import torchvision.transforms as tfs
from dataclasses import dataclass
import datasets as hfds
import huggingface_hub as hfhub
from .common import BaseFixScore


HF_DATA_REPO = "BrachioLab/cholec"


class CholecDataset(Dataset):
    """
    The cholecystectomy (gallbladder surgery) dataset, loaded from HuggingFace.
    The task is to find the safe/unsafe (gonogo) regions.
    The expert-specified features are the organ labels.

    For more details, see: https://huggingface.co/datasets/BrachioLab/cholecystectomy
    """

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
        r"""
        Args:
            split: The options are "train" and "test".
            hf_data_repo: The HuggingFace repository where the dataset is stored.
            image_size: The (height, width) of the image to load.
        """
        self.dataset = hfds.load_dataset(hf_data_repo, split=split)
        self.dataset.set_format("torch")
        self.image_size = image_size
        self.preprocess_image = tfs.Compose([
            tfs.Lambda(lambda x: x.float() / 255),
            tfs.Resize(image_size), # for datasets version too old, the dimension can be (3, H, W) and this will break
        ])
        self.preprocess_labels = tfs.Compose([
            tfs.Lambda(lambda x: x.unsqueeze(0)),
            tfs.Resize(image_size),
            tfs.Lambda(lambda x: x[0])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset[idx]['image'].shape[:2] == self.image_size:
            image = self.dataset[idx]['image'].permute(2,0,1)
        else:
            image = self.dataset[idx]['image']
        image = self.preprocess_image(image)
        gonogo = self.preprocess_labels(self.dataset[idx]["gonogo"]).long()
        organs = self.preprocess_labels(self.dataset[idx]["organ"]).long()
        return {
            "image": image,     # (3,H,W)
            "gonogo": gonogo,   # (H,W)
            "organs": organs,   # (H,W)
        }


@dataclass
class CholecModelOutput:
    logits: torch.FloatTensor


class CholecModel(nn.Module, hfhub.PyTorchModelHubMixin):
    """
    Loads an image segmentation model for either: gonogo segmentation, or organ segmentation.
    The PyTorchModelHubMixin is what lets us do convenient upload/download from HuggingFace
    using `CholecModel.from_pretrained("BrachioLab/cholecystectomy_gonogo")`

    For more details, see:
    * https://huggingface.co/BrachioLab/cholecystectomy_organs
    * https://huggingface.co/BrachioLab/cholecystectomy_gonogo
    """
    def __init__(self, task: str = "gonogo"):
        r"""
        Args:
            task: Either "gonogo" or "organs"
        """
        super().__init__()
        self.task = task
        if task == "gonogo":
            self.num_labels = 3
        elif task == "organs":
            self.num_labels = 4
        else:
            raise ValueError(f"Unrecognized task {task}")

        self.seg_model = tvm.segmentation.fcn_resnet50(num_classes=self.num_labels)

        # Normalization numbers should be custom computed, but we'll just use ImageNet's :)))
        self.preprocess = tfs.Compose([
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x: torch.FloatTensor):
        x = self.preprocess(x)
        seg_out = self.seg_model(x)
        return CholecModelOutput(
            logits = seg_out["out"]
        )


class CholecFixScore(BaseFixScore):
    """
    The FIX score for CholecDataset, where the explicit expert features are known.
    """
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
        Args:
            groups_pred: A binary-valued tensor of shape (N,P,H W), where P is the number of predicted groups
            groups_true: A binary-valued tensor of shape (N,T,H,W), where T is the number of true groups
        """
        N, P, H, W = groups_pred.shape
        _, T, H, W = groups_true.shape

        # Make sure to binarize groups and shorten names to help with math
        Gp = groups_pred.bool().long()
        Gt = groups_true.bool().long()

        # Make (N,P,T)-shaped lookup tables for the intersection and union
        if big_batch:
            # This is the more flashy PyTorch-esque way, but it's not very memory-efficient,
            # Because the effective batch size is N*P*T, which might be quite big
            inters = (Gp.view(N,P,1,H,W) * Gt.view(N,1,T,H,W)).sum(dim=(-1,-2))
            unions = (Gp.view(N,P,1,H,W) + Gt.view(N,1,T,H,W)).clamp(0,1).sum(dim=(-1,-2))
        else:
            # Uses for-loops, but is far more memory-efficient.
            inters = torch.zeros(N,P,T).to(Gp.device)
            unions = torch.zeros(N,P,T).to(Gp.device)
            for i in range(P):
                for j in range(T):
                    inters[:,i,j] = (Gp[:,i] * Gt[:,j]).sum(dim=(-1,-2))
                    unions[:,i,j] = (Gp[:,i] + Gt[:,j]).clamp(0,1).sum(dim=(-1,-2))
        ious = inters / unions  # (N,P,T)
        ious[~ious.isfinite()] = 0 # Set nans and inftys to zero.
        iou_maxs = ious.max(dim=-1).values   # (N,P): max_{gt in Gt} iou(gp, gt)

        # sum_{gp in group_preds(feature)} iou_max(gp, Gt)
        pred_aligns_sum = (Gp * iou_maxs.view(N,P,1,1)).sum(dim=1) # (N,H,W)
        score = pred_aligns_sum / Gp.sum(dim=1) # (N,H,W), division is the |Gp(feaure)|
        score[~score.isfinite()] = 0    # Make div-by-zero things zero
        if reduce:
            return score.mean(dim=(1,2))
        else:
            return score    # (N,H,W), a score for each feature


r"""
Some code for running the FIX score on different baselines.
"""

_all_cholec_baselines = [
    'identity',
    'random',
    'patch',
    'quickshift',
    'watershed',
    'sam',
    'ace',
    'craft',
    'archipelago'
]

def get_cholec_scores(
    baselines = _all_cholec_baselines,
    num_todo = None,
    batch_size = 8,
    device = "cuda" if torch.cuda.is_available() else "cpu",
):
    from tqdm import tqdm
    import sys
    sys.path.append("../../..")
    import exlib.features.vision as xfv

    torch.manual_seed(1234)
    dataset = CholecDataset(split="test")
    metric = CholecFixScore()

    if num_todo is not None:
        dataset = Subset(dataset, torch.randperm(len(dataset))[:num_todo].tolist())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    resizer = tfs.Resize((180,320)) # A smaller image makes algos like quickshift much faster.
    
    all_baselines_scores = {}
    for item in tqdm(dataloader):
        for baseline in baselines:
            if baseline == "identity":
                groups = xfv.IdentityGroups()
            elif baseline == "random":
                groups = xfv.RandomGroups(max_groups=8)
            elif baseline == "patch": # patch
                groups = xfv.PatchGroups(grid_size=(8,14), mode="grid")
            elif baseline == "quickshift": # quickshift
                groups = xfv.QuickshiftGroups(max_groups=8)
            elif baseline == "watershed": # watershed
                groups = xfv.WatershedGroups(max_groups=8)
            elif baseline == "sam": # watershed
                groups = xfv.SamGroups(max_groups=8)
            elif baseline == "ace":
                groups = xfv.NeuralQuickshiftGroups(max_groups=8)
            elif baseline == "craft":
                groups = xfv.CraftGroups(max_groups=8)
            elif baseline == "archipelago":
                groups = xfv.ArchipelagoGroups(max_groups=8)
            else:
                raise ValueError(f"Unknown baseline {baseline}")

            groups.eval().to(device)

            image = resizer(item["image"].to(device))

            with torch.no_grad():
                organ_masks = F.one_hot(item["organs"]).permute(0,3,1,2).to(device)
                organ_masks = resizer(organ_masks.float()).long()
                pred_masks = groups(image)
                score = metric(pred_masks, organ_masks).cpu() # (N,H,W)

                if baseline in all_baselines_scores.keys():
                    scores = all_baselines_scores[baseline]
                    scores.append(score)
                else: 
                    scores = [score]
                all_baselines_scores[baseline] = scores

    for baseline in baselines:
        scores = torch.cat(all_baselines_scores[baseline])
        all_baselines_scores[baseline] = scores

    return all_baselines_scores
    
# get preprocessed data for input into feature extractor and metric    
def preprocess_cholec(batch):
    import torchvision.transforms as tfs
    resizer = tfs.Resize((180,320))
    x = resizer(batch['image'])
    organ_masks = F.one_hot(batch["organs"]).permute(0,3,1,2)
    organ_masks = resizer(organ_masks.float()).long()
    X = {'x': x}
    metric_inputs = {'groups_true': organ_masks}
    return X, metric_inputs
    
