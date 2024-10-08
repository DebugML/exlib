import torch
from torch.utils.data import Dataset
from torchvision import transforms
import datasets as hfds


HF_DATA_REPO = "BrachioLab/visa"


class VisADataset(Dataset):

    categories = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum"
    ]

    def __init__(
        self,
        category,
        split,
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

