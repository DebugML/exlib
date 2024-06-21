import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import PretrainedConfig, PreTrainedModel
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from datasets import load_dataset
from collections import defaultdict
import torch.nn.functional as F
from tqdm.auto import tqdm
import sys
sys.path.append("../src")
import exlib
# Baselines
from exlib.features.vision.mass_maps import MassMapsOracle, MassMapsOne
from exlib.features.vision.watershed import WatershedGroups
from exlib.features.vision.quickshift import QuickshiftGroups
from exlib.features.vision.patch import PatchGroups


DATASET_REPO = "BrachioLab/massmaps-cosmogrid-100k"
MODEL_REPO = "BrachioLab/massmaps-conv"

"""
To use dataset
from datasets import load_dataset
from exlib.datasets import mass_maps
dataset = load_dataset(mass_maps.DATASET_REPO)

To use model
from exlib.datasets.mass_maps import MassMapsConvnetForImageRegression
model = MassMapsConvnetForImageRegression.from_pretrained(exlib.datasets.mass_maps.MODEL_REPO)

To use metrics
from exlib.datasets.mass_maps import MassMapsMetrics
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
                 eps=1e-6,
                #  void_scale=1, cluster_scale=1
                 ):
        super().__init__()
        self.void_threshold = void_threshold
        self.cluster_threshold = cluster_threshold
        self.eps = eps

    def forward(self, groups, x, reduce='sum'):
        """
        group: (N, M, H, W) 0 or 1, or bool
        x: image (N, 1, H, W)
        return 
        """
        assert reduce in ['sum', 'none']
        # metric test
        groups = groups.bool().float() # if float then turned into bool
        masked_imgs = groups * x # (N, M, H, W)
        sigma = x.flatten(2).std(dim=-1) # (N, M)
        mask_masses = (masked_imgs * groups).flatten(2).sum(-1)
        img_masses = groups.flatten(2).sum(-1)
        mask_intensities = mask_masses / img_masses
        mask_sizes = groups.flatten(2).sum(-1) # (N, M)
        num_masks = mask_sizes.bool().sum(-1) # (N, M)
        
        p_void = (masked_imgs < self.void_threshold*sigma[:,:,None,None]).sum([-1,-2]) / mask_sizes 
        p_cluster = (masked_imgs > self.cluster_threshold*sigma[:,:,None,None]).sum([-1,-2]) / mask_sizes
        p_other = 1 - p_void - p_cluster

        p_void_ = (p_void + self.eps) / (p_void + p_cluster + p_other + self.eps * 3)
        p_cluster_ = (p_cluster + self.eps) / (p_void + p_cluster + p_other + self.eps * 3)
        p_other_ = (p_other + self.eps) / (p_void + p_cluster + p_other + self.eps * 3)

        p_void_vconly = p_void_ / (p_void_ + p_cluster_)
        p_cluster_vconly = p_cluster_ / (p_void_ + p_cluster_)

        purity = 1/2 * (2 + p_void_vconly * torch.log2(p_void_vconly) + \
            p_cluster_vconly * torch.log2(p_cluster_vconly)) * (p_void_ + p_cluster_) # (N, M)

        # align_i
        purity[mask_sizes.bool().logical_not()] = 0 
        alignment = (purity[:,:,None,None] * groups).sum(1) / groups.sum(1)
        alignment[groups.sum(1) == 0] = 0

        if reduce == 'sum':
            return alignment # (N, H, W)
        else: # none
            return {
                'alignment': alignment,
                'purity': purity,
                'p_void_vconly': p_void_vconly,
                'p_cluster_vconly': p_cluster_vconly,
                'p_void_': p_void_,
                'p_cluster_': p_cluster_,
                'p_other_': p_other_
            }


def show_example(groups, X, img_idx=0, mode='contour'):
    assert mode in ['contour', 'dim']
    massmaps_align = MassMapsAlignment()
    alignment_results = massmaps_align(groups, X, reduce='none')
    
    m = groups.shape[1]
    cols = 8
    rows = math.ceil(m / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*4))
    axs = axs.ravel()

    image = X[img_idx]
    for idx in range(len(axs)):
        if idx < m:
            mask = groups[img_idx][idx]

            if mask.sum() > 0:
                axs[idx].imshow(image[0].cpu().numpy())
                axs[idx].contour(mask.cpu().numpy() > 0, 2, colors='red')
                axs[idx].contourf(mask.cpu().numpy() > 0, 2, hatches=['//', None, None],
                                cmap='gray', extend='neither', linestyles='--', alpha=0.01)
                p_void_ = alignment_results['p_void_']
                p_cluster_ = alignment_results['p_cluster_']
                purity = alignment_results['purity']
                # total_score = alignment_scores_void[0][idx].item() * massmaps_align.void_scale + alignment_scores_cluster[0][idx].item() * massmaps_align.cluster_scale
                axs[idx].set_title(f'void {p_void_[0][idx].item():.5f}\ncluster {p_cluster_[0][idx].item():.5f}\npurity {purity[0][idx].item():.5f}')
        axs[idx].axis('off')
    plt.show()



def map_plotter(image, mask, ax=plt, type='dim'): 
    default_cmap = matplotlib.rcParams['image.cmap']
    cmap = plt.get_cmap(default_cmap)
    gray_cmap = plt.get_cmap('gray')
    
    norm = plt.Normalize()
    normed_image = norm(image)
    # normed_image = np.clip((image - img_min) / (img_max - img_min) * 3, 0, 1)
    rgb_image = cmap(normed_image)
    if type == 'dim':
        gray_image = (gray_cmap(normed_image) / 2)
        mask_bool = np.repeat((mask > 0)[:,:,None], 4, axis=-1)
        gray_image[mask_bool] = rgb_image[mask_bool]
        ax.imshow(gray_image)
    elif type == 'contour':
        # 1. Find the contours of the mask
        mask_bool = (mask > 0).astype(np.float32)
        dilated_mask_bool = binary_dilation(mask_bool)
        contours_original = measure.find_contours(mask_bool, 0.5)
        contours_dilated = measure.find_contours(dilated_mask_bool, 0.51)

        # 2. Overlay the contours on top of the original image
        # fig, ax = plt.subplots()
        ax.imshow(rgb_image)

        for contour in contours_dilated:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='white')  # color can be changed as needed

        for contour in contours_original:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')  # color can be changed as needed



def get_mass_maps_scores(baselines = ['patch', 'quickshift', 'watershed', 'oracle', 'one'], subset=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load model
    model = MassMapsConvnetForImageRegression.from_pretrained(MODEL_REPO) # BrachioLab/massmaps-conv
    model = model.to(device)
    
    # Load data
    train_dataset = load_dataset(DATASET_REPO, split='train') # BrachioLab/massmaps-cosmogrid-100k
    val_dataset = load_dataset(DATASET_REPO, split='validation')
    test_dataset = load_dataset(DATASET_REPO, split='test')
    train_dataset.set_format('torch', columns=['input', 'label'])
    val_dataset.set_format('torch', columns=['input', 'label'])
    test_dataset.set_format('torch', columns=['input', 'label'])
    
    massmaps_align = MassMapsAlignment()
    
    batch_size = 16
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    mse_loss_all = 0
    total = 0
    # alignment_scores_all = [[] for _ in range(len(baselines))]
    all_baselines_scores = {}
    
    with torch.no_grad():
        for bi, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            if subset and bi % 100 != 0:
                continue
            X = batch['input'].to(device)
            y = batch['label'].to(device)
            out = model(X)
            # loss
            loss = F.mse_loss(out, y, reduction='none')
            mse_loss_all = mse_loss_all + loss.sum(0)
            total += X.shape[0]
    
            # baseline
            for base_i, baseline_name in enumerate(baselines):
                if baseline_name == 'patch':
                    baseline = PatchGroups(num_patches=(8, 8), mode='count')
                elif baseline_name == 'quickshift':
                    baseline = QuickshiftGroups(kernel_size=5, max_dist=10)
                elif baseline_name == 'watershed':
                    baseline = WatershedGroups(min_dist=10, compactness=0)
                elif baseline_name =='oracle':
                    baseline = MassMapsOracle()
                elif baseline_name =='one':
                    baseline = MassMapsOne()
                else:
                    raise Exception("Please indicate a valid baseline")
                
                groups = baseline(X)
    
                # alignment
                scores_batch = massmaps_align(groups, X)
                scores_batch = scores_batch.flatten(1).cpu().numpy().tolist()

                if baseline_name in all_baselines_scores.keys():
                    scores = all_baselines_scores[baseline_name]
                    scores = scores + scores_batch
                else: 
                    scores = scores_batch
                all_baselines_scores[baseline_name] = scores
                                
    loss_avg = mse_loss_all / total
    print(f'Omega_m loss {loss_avg[0].item():.4f}, sigma_8 loss {loss_avg[1].item():.4f}, avg loss {loss_avg.mean().item():.4f}')

    # print(all_baselines_scores)
    for baseline_name in baselines:
        scores = torch.tensor(all_baselines_scores[baseline_name])
        # print(f"Avg alignment of {baseline_name} features: {scores.mean():.4f}")
        all_baselines_scores[baseline_name] = scores

    return all_baselines_scores