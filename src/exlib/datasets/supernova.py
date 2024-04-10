import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob
from collections import namedtuple
from typing import Optional, Union, List

import pyarrow as pa
import pyarrow_hotfix
from informer_models import InformerForSequenceClassification
from datasets import load_dataset
import torch
import yaml
from collections import namedtuple

from connect_later.dataset_preprocess_raw import create_train_dataloader_raw, create_test_dataloader_raw
from connect_later.informer_models import InformerForSequenceClassification
from connect_later.pretrain import get_dataset, setup_model_config

class Supernova(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        dataset = load_dataset(data_dir, trust_remote_code=True)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data

    def __repr__(self):
        if self.split:
            features = list(self.dataset.features.keys())
            num_rows = len(self.dataset)
            return f"DatasetDict({{\n    {self.split}: Dataset({{\n        features: {features},\n        num_rows: {num_rows}\n    }})\n}})"
        else:
            return str(self.dataset)

    def rename_column(self, old_name, new_name):
        if old_name in self.dataset.features:
            self.dataset = self.dataset.rename_column(old_name, new_name)
        else:
            raise ValueError(f"Column '{old_name}' does not exist in the dataset.")
            
class SupernovaClsModel(nn.Module):
    def __init__(self, model_path, config_path):
        super(SupernovaClsModel, self).__init__()
        state_dict = torch.load(model_path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        Args = namedtuple('Args', ['has_labels', 'num_labels', 'regression', 'classifier_dropout', 'fourier_pe', 'mask', 'mask_probability'])

        args = Args(
            has_labels=True,
            num_labels=14,
            regression=False,
            classifier_dropout=0.2,
            fourier_pe=True,
            mask=True,
            mask_probability=0.6
        )
        
        finetune_config = {
            "has_labels": True,
            "num_labels": 14,
            "regression": False,
            "classifier_dropout": 0.2,
            "fourier_pe": True,
            "mask": True
        }
        self.model_config = setup_model_config(args, config)
        self.model_config.update(finetune_config)
        self.model = InformerForSequenceClassification(self.model_config)
        self.model.load_state_dict(state_dict['model_state_dict'])

    def forward(self, *input, **kwargs):
        return self.model(*input, **kwargs)