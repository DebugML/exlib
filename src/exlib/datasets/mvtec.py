import torch
from torch.utils.data import Dataset
from torchvision import transforms
import datasets as hfds


HF_DATA_REPO = "BrachioLab/mvtec-ad"

class MVTecDataset(Dataset):

    categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    def __init__(
        self,
        category: str,
        split: str,
        image_size: int = 256,
        hf_data_repo = HF_DATA_REPO,
    ):
        self.split = split
        self.dataset = hfds.load_dataset(hf_data_repo, split=(category + "." + split))
        self.dataset.set_format("torch")
        self.resize = transforms.Resize(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.resize(item["image"])

        if self.split == "train":
            return {
                "image": image,
                "mask": torch.zeros_like(image).long(),
                "label": 0
            }

        else:
            return {
                "image": image,
                "mask": (self.resize(item["mask"]) > 0).long(),
                "label": item["label"]
            }

