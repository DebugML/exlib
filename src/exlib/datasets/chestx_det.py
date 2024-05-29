import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchxrayvision as xrv


class ChestXDet(Dataset):
    disease_names: str = [
        "Atelectasis",
        "Calcification",
        "Cardiomegaly",
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
        data_dir,
        split: str = "train",
        image_size: int = 1024,
    ):
        self.data_dir = data_dir
        assert split in ["train", "test"]
        self.split = split
        self.images_dir = os.path.join(data_dir, "data", split, "images")
        self.struct_labels_dir = os.path.join(data_dir, "data", split, "structure_labels")
        self.disease_labels_dir = os.path.join(data_dir, "data", split, "disease_labels")

        assert os.path.isdir(self.images_dir)
        assert os.path.isdir(self.struct_labels_dir)
        assert os.path.isdir(self.disease_labels_dir)
        self.image_ids = sorted([f.split(".")[0] for f in os.listdir(self.images_dir)])
        self.image_size = image_size
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size)
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_file = os.path.join(self.images_dir, image_id + ".png")
        image = self.transforms(Image.open(image_file))

        disease_labels = torch.zeros(13, self.image_size, self.image_size)
        for i in range(13):
            di_file = os.path.join(self.disease_labels_dir, image_id + f"_disease_{i}.png")
            if os.path.isfile(di_file):
                di_label = self.transforms(Image.open(di_file).convert("L"))
                disease_labels[i] = di_label[0]

        struct_labels = torch.zeros(14, self.image_size, self.image_size)
        for i in range(14):
            si_file = os.path.join(self.struct_labels_dir, image_id + f"_structure_{i}.png")
            if os.path.isfile(si_file):
                si_label = self.transforms(Image.open(si_file).convert("L"))
                struct_labels[i] = si_label[0]

        return image.float(), disease_labels.long(), struct_labels.long()



class PathologySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.xrv_model = xrv.models.DenseNet(weights="densenet121-res224-nih")

    def forward(self, x):
        return self.xrv_model(x)


class StructureSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.xrv_model = xrv.baseline_models.chestx_det.PSPNet()

    def forward(self, x: torch.Tensor):
        img_size = x.shape[2:]
        x = 2048 * (x - 0.5) # Scale form [0,1] to [-1024, 1024]
        x = x.mean(dim=0, keepdim=True) # (-1,1,H,W), Aggregate along color channels, if any
        xrv_out = self.xrv_model(x)
        out = F.upsample(xrv_out, size=img_size)
        return out

