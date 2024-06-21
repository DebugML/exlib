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
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from exlib.features.time_series.chunk import BaselineGroups
from datasets import load_dataset
import torch
import yaml
from collections import namedtuple
import huggingface_hub as hfhub

from exlib.utils.supernova_helper import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SupernovaDataset(Dataset):
    def __init__(self, data_dir = "BrachioLab/supernova-timeseries", config_path = "BrachioLab/supernova-classification", split: str = "test"):
        self.dataset = load_dataset(data_dir, split = split)
        self.config = InformerConfig.from_pretrained(config_path)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
        
            
class SupernovaClsModel(nn.Module):
    def __init__(self, model_path = "BrachioLab/supernova-classification"):
        super().__init__()
        self.model = InformerForSequenceClassification.from_pretrained(model_path)

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return out

def calculate_means(x, masks):
    x, masks = x.to(device), masks.to(device)
    x_sums = torch.matmul(masks.float(), x[:,None,:,None]).squeeze(3)
    x_totals = masks.sum(-1)
    x_totals = torch.clamp(x_totals, min=1e-5)
    return x_sums / x_totals

class SupernovaFIXScores(nn.Module): 
    def __init__(self, sigma = 1, nchunk = 7, groups=torch.Tensor([]), labels=None, past_values=None, past_time_features=None, past_observed_mask=None): 
        super(SupernovaFIXScores, self).__init__()
        self.sigma = sigma
        self.nchunk = nchunk
        self.groups = groups

    def forward(self, labels=None, past_values=None, past_time_features=None, past_observed_mask=None):
        sigma, nchunk, groups = self.sigma, self.nchunk, self.groups
        t, wl, flux, err = past_time_features[:,:,0].to(device), past_time_features[:,:,1].to(device), past_values[:,:,0].to(device), past_values[:,:,1].to(device)
        unique_wl = torch.Tensor([3670.69, 4826.85, 6223.24, 7545.98, 8590.9, 9710.28]).to(device)
        
        t_means = calculate_means(t, groups)
        flux_means = calculate_means(flux, groups)
        t_delta = t[:,None,None,:] - t_means.unsqueeze(3)
        flux_delta = flux[:,None,None,:] - flux_means.unsqueeze(3)
        
        actual_batch_size = wl.size(0)
        groups = groups[:actual_batch_size].to(device)
        groups_by_wl = (wl[:,None,None,:] == unique_wl[:,None]) & (groups.unsqueeze(2))

        t_means = calculate_means(t, groups_by_wl)
        flux_means = calculate_means(flux, groups_by_wl)
        t_delta = t[:,None,None,:] - t_means.unsqueeze(3)
        flux_delta = flux[:,None,None,:] - flux_means.unsqueeze(3)
        
        # weighted linear regression by groups
        numerator = (groups_by_wl*t_delta*flux_delta).sum(-1)
        denominator = (groups_by_wl*t_delta*t_delta).sum(-1)
        beta = numerator / denominator
        alpha = flux_means - t_means*beta
        flux_pred = t[:,None,None,:]*beta.unsqueeze(3) + alpha.unsqueeze(3)
        flux_diff = (flux_pred - flux[:,None,None,:]).abs()
        flux_pass = flux_diff <= sigma*err[:,None,None,:]
        percent_flux_pass = (flux_pass*groups_by_wl).sum(-1) / groups_by_wl.sum(-1) # batch x group x wl
        
        groups_t = t[:,None,:] * groups
        groups_t_lower = groups_t.min(dim=2).values
        groups_t_upper = groups_t.max(dim=2).values

        chunk_step_size = (groups_t_upper - groups_t_lower)/nchunk
        chunk_window_size = chunk_step_size * 2
        chunk_steps = torch.arange(0,nchunk,device=groups_t.device)*(chunk_step_size.unsqueeze(2))
        chunk_time_start = groups_t_lower[:,:,None] + chunk_steps
        chunk_time_end = torch.min(chunk_time_start + chunk_window_size[:,:,None], groups_t_upper[:,:,None])
        chunk_groups = (chunk_time_start[:,:,:,None] <= t[:,None,None,:]) & (chunk_time_end[:,:,:,None] >= t[:,None,None,:])
        chunk_groups_by_wl = (wl[:,None,None,None,:] == unique_wl[None,:,None]) & (chunk_groups.unsqueeze(3))
        chunk_sum = chunk_groups_by_wl.sum(-1).transpose(2, 3)
        percent_chunk_pass = torch.count_nonzero(chunk_sum, dim=3) / nchunk
        
        alignment_score_wv = percent_flux_pass * percent_chunk_pass
        alignment_score, _ = torch.max(alignment_score_wv, dim=2)
        alignment_score = torch.nan_to_num(alignment_score)
        scores_per_feature = (alignment_score.unsqueeze(2)*groups)
        
        return (scores_per_feature.sum(1)/torch.clamp(groups.sum(1), min=1e-5)).mean(dim=1)

def get_supernova_scores(
    baselines = ['chunk 5', 'chunk 10', 'chunk 15'],
    dataset = SupernovaDataset(data_dir = "BrachioLab/supernova-timeseries", split="test"),
    SupernovaFIXScore = SupernovaFIXScores(sigma=1, nchunk=7, groups = torch.Tensor([])),
    batch_size = 5,
):
    test_dataloader = create_test_dataloader_raw(
    dataset=dataset,
    batch_size=5,
    compute_loss=True
    )
    
    all_baselines_scores = {}
    with torch.no_grad():
        for bi, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            for baseline in baselines:
                if baseline == 'chunk 5':
                    BaselineGroup = BaselineGroups(ngroups=5, window_size=100)
                elif baseline == 'chunk 10':
                    BaselineGroup = BaselineGroups(ngroups=10, window_size=100)
                elif baseline == 'chunk 15':
                    BaselineGroup = BaselineGroups(ngroups=15, window_size=100)
                else:
                    raise Exception("Please indicate a valid baseline")

                pred_groups = BaselineGroup(**batch)
                SupernovaFIXScore = SupernovaFIXScores(sigma=1, nchunk=7, groups = pred_groups)
                scores_batch = SupernovaFIXScore(**batch)
                # print(scores_batch)

                if baseline in all_baselines_scores.keys():
                    scores = all_baselines_scores[baseline]
                    scores = scores + scores_batch.tolist()
                else: 
                    scores = scores_batch.tolist()
                all_baselines_scores[baseline] = scores
                
    # print(all_baselines_scores)
    for baseline in baselines:
        scores = torch.tensor(all_baselines_scores[baseline])
        # print(f"Avg alignment of {baseline} features: {scores.mean():.4f}")
        all_baselines_scores[baseline] = scores
        
    return all_baselines_scores

def plot_data_by_wavelength(times, fluxes, errors, wavelengths, title, bi, j):
    unique_wavelengths = sorted(set(wavelengths))
    color_map = plt.get_cmap('rainbow')
    colors = color_map(np.linspace(0, 1, len(unique_wavelengths)))
    wavelength_to_color = {w: c for w, c in zip(unique_wavelengths, colors)}

    plt.figure(figsize=(4, 4))
    for wavelength in unique_wavelengths:
        indices = [i for i, w in enumerate(wavelengths) if w == wavelength]
        plt.errorbar([times[i] for i in indices], [fluxes[i] for i in indices], yerr=[errors[i] for i in indices],
                     fmt='o', color=wavelength_to_color[wavelength], capsize=5, label=f'{int(wavelength)}')
    plt.xlabel('Time')
    plt.ylabel('Flux')
    #plt.title(title, fontsize=10)
    
    plt.legend(title="Wavelengths", loc='upper right', fontsize='small', title_fontsize='small')
    #plt.savefig(f'groups_example/plot_org_{bi}_{j}.png', format='png', dpi=300, bbox_inches='tight')
    plt.grid(False)
    plt.show()
