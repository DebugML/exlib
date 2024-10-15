import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as tfs
from dataclasses import dataclass
import torchxrayvision as xrv
import datasets as hfds
import huggingface_hub as hfhub
from .common import BaseFixScore


HF_DATA_REPO = "BrachioLab/chestx"


class ChestXDataset(Dataset):
    """
    The ChestX-ray dataset adapted from torchxrayvision's curation of the GoogleNIH dataset.
    The structure segmentations are also derived using torchxrayvision's off-the-shelf models.

    This dataset is hosted on HuggingFace, see:
    https://huggingface.co/datasets/BrachioLab/chestx
    """

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
        r"""
        Args:
            split: Either "train" or "test"
            hf_data_repo: Where the dataset is hosted on HuggingFace
            image_size: The square image size
        """
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
        image = self.dataset[idx]["image"]
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        image = self.preprocess_image(image)
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
        """
        Args:
            x: (N,C,224,224) with values in [0,1], with either C=1 or C=2 channels
        """

        x = x * 2048 - 1024 # The xrv model requires some brazingo scaling
        out = self.xrv_model(x)
        """
        The torchxrayvision model outputs 18 pathology labels in the following order:
            [
                'Atelectasis',
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
                 ''
            ]
        ... so we need to sort it to match our ordering
        """
        pathol_idxs = [0, 10, 1, 4, 7, 5, 6, 13, 2, 12, 11, 9, 8, 3]
        return out[:,pathol_idxs]


class ChestXFixScore(BaseFixScore):
    """
    The FIX score for ChestXDataset, where the explicit expert features are known.
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


r"""
Some code for running the FIX score on different baselines.
"""

_all_chestx_baselines = [
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


def get_chestx_scores(
    baselines = _all_chestx_baselines,
    num_todo = 256,
    batch_size = 16,
    device = "cuda" if torch.cuda.is_available() else "cpu",
):

    from tqdm import tqdm
    import sys
    sys.path.append("../..")
    import exlib.features.vision as xfv

    torch.manual_seed(1234)
    dataset = ChestXDataset(split="test")
    metric = ChestXFixScore()

    if num_todo is not None:
        dataset = Subset(dataset, torch.randperm(len(dataset))[:num_todo].tolist())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    all_baselines_scores = {}
    for item in tqdm(dataloader):
        for baseline in baselines:
            if baseline == "identity":
                groups = xfv.IdentityGroups()
            elif baseline == "random":
                groups = xfv.RandomGroups(max_groups=20)
            elif baseline == "patch": # patch
                groups = xfv.PatchGroups(grid_size=(8,8), mode="grid")
            elif baseline == "quickshift": # quickshift
                groups = xfv.QuickshiftGroups(max_groups=20)
            elif baseline == "watershed": # watershed
                groups = xfv.WatershedGroups(max_groups=20)
            elif baseline == "sam":
                groups = xfv.SamGroups(max_groups=20)
            elif baseline == "ace":   # ACE
                groups = xfv.NeuralQuickshiftGroups(max_groups=20)
            elif baseline == "craft":
                groups = xfv.CraftGroups(max_groups=20)
            elif baseline == "archipelago":
                groups = xfv.ArchipelagoGroups(max_groups=20)
            else:
                raise ValueError(f"Unknown baseline {baseline}")

            groups.eval().to(device)

            image = item["image"].to(device)
            if baseline == "archipelago":
                image = image.repeat(1,3,1,1)

            with torch.no_grad():
                structs_masks = item["structs"]
                pred_masks = groups(image)

                structs_masks = structs_masks.to(device)
                pred_masks = pred_masks.to(device)
                score = metric(pred_masks, structs_masks) # (N,H,W)

                if baseline in all_baselines_scores.keys():
                    scores = all_baselines_scores[baseline]
                    scores.append(score) #.mean(dim=(1,2)))
                else: 
                    scores = [score] #.mean(dim=(1,2))]
                all_baselines_scores[baseline] = scores

    for baseline in baselines:
        scores = torch.cat(all_baselines_scores[baseline])
        all_baselines_scores[baseline] = scores

    return all_baselines_scores
        

def preprocess_chestx(batch):
    x = batch['image'] # might need image = image.repeat(1,3,1,1) for archipelago
    structs_masks = batch["structs"]
    X = {'x': x}
    metric_inputs = {'groups_true': structs_masks}
    return X, metric_inputs
