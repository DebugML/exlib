import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import PretrainedConfig, PreTrainedModel

DATASET_REPO = "BrachioLab/massmaps-cosmogrid-100k"
MODEL_REPO = "BrachioLab/massmaps-conv"

"""
To use dataset
from datasets import load_dataset
from exlib.datasets import massmaps
dataset = load_dataset(massmaps.DATASET_REPO)

To use model
from exlib.datasets.massmaps import MassMapsConvnetForImageRegression
model = MassMapsConvnetForImageRegression.from_pretrained(exlib.datasets.massmaps.MODEL_REPO)

To use metrics
from exlib.datasets.massmaps import MassMapsMetrics
massmaps_metrics = MassMapsMetrics()
massmaps_metrics(zp, x)
"""

class ModelOutput:
    def __init__(self, logits, pooler_output, hidden_states=None):
        self.logits = logits
        self.pooler_output = pooler_output
        self.hidden_states = hidden_states


class MassMapsConvnetConfig(PretrainedConfig):
    def __init__(self, 
        num_classes=2, 
        **kwargs
        ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        
class MassMapsConvnetForImageRegression(PreTrainedModel):
    config_class = MassMapsConvnetConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        num_classes = config.num_classes
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4)
        self.relu1 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4)
        self.relu2 = nn.LeakyReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=4)
        self.relu3 = nn.LeakyReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1200, 128)
        self.relu4 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu5 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu6 = nn.LeakyReLU()
        self.fc4 = nn.Linear(32, num_classes)
        
    def forward(self, 
                x: torch.Tensor,
                output_hidden_states: Optional[bool] = False, 
                return_dict: Optional[bool] = False):
        """
        x: torch.Tensor (batch_size, 1, 66, 66)
        output_hidden_states: bool
        return_dict: bool
        """
        hidden_states = []
        x = self.conv1(x)
        if output_hidden_states:
            hidden_states.append(x.clone())
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        if output_hidden_states:
            hidden_states.append(x.clone())
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        if output_hidden_states:
            hidden_states.append(x.clone())
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        pooler_output = self.relu6(x)
        logits = self.fc4(pooler_output)

        if not return_dict:
            return logits
        
        return ModelOutput(
            logits=logits,
            pooler_output=pooler_output,
            hidden_states=hidden_states
        )


class MassMapsAlignment(nn.Module):
    def __init__(self, void_threshold=0, cluster_threshold=3, 
                 void_scale=1, cluster_scale=1):
        super().__init__()
        self.void_threshold = void_threshold
        self.cluster_threshold = cluster_threshold
        self.daf_types = ['other', 'void', 'cluster']
        self.daf_types2id = {
            'other': 0,
            'void': 1,
            'cluster': 2
        }
        self.void_scale = void_scale
        self.cluster_scale = cluster_scale
        
    def forward(self, zp, x, reduce='sum'):
        """
        zp: (N, M, H, W) 0 or 1, or bool
        x: image (N, 1, H, W)
        return 
        """
        assert reduce in ['sum', 'none']
        # metric test
        zp = zp.bool().float() # if float then turned into bool
        masked_imgs = zp * x # (N, M, H, W)
        sigma = x.flatten(2).std(dim=-1) # (N, M)
        mask_masses = (masked_imgs * zp).flatten(2).sum(-1)
        img_masses = zp.flatten(2).sum(-1)
        mask_intensities = mask_masses / img_masses
        mask_sizes = zp.flatten(2).sum(-1) # (N, M)
        num_masks = mask_sizes.bool().sum(-1) # (N, M)
        
        align_void = (masked_imgs < self.void_threshold*sigma[:,:,None,None]).sum([-1,-2]) \
                        / mask_sizes * (- mask_masses)
        align_void[mask_sizes.bool().logical_not()] = 0
        align_void[align_void < 0] = 0
        align_cluster = (masked_imgs > self.cluster_threshold*sigma[:,:,None,None]).sum([-1,-2]) \
                        / mask_sizes
        align_cluster[mask_sizes.bool().logical_not()] = 0
        align_cluster[align_cluster < 0] = 0
        
        if reduce == 'sum':
            return align_void * self.void_scale + align_cluster * self.cluster_scale
        else: # none
            return align_void, align_cluster
