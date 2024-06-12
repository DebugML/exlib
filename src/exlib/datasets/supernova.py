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

from informer_models import InformerForSequenceClassification
from datasets import load_dataset
import torch
import yaml
from collections import namedtuple

from exlib.datasets.pretrain import setup_model_config, get_dataset, get_dataset, setup_model_config
from exlib.datasets.dataset_preprocess_raw import create_train_dataloader_raw, create_test_dataloader_raw, create_test_dataloader
from exlib.datasets.informer_models import InformerConfig, InformerForSequenceClassification

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


def baseline(valid_length):
    chunk_size = 30
    num_groups = valid_length // chunk_size
    if valid_length % chunk_size != 0:
        num_groups += 1
    pred_groups = []
    for group_idx in range(num_groups):
        start_index = group_idx * chunk_size
        end_index = min((group_idx + 1) * chunk_size, valid_length)
        group_list = [1 if start_index <= i < end_index else 0 for i in range(valid_length)]
        pred_groups.append(group_list)
    return pred_groups

def plot_flux_time_with_error_bars_group(times, wavelengths, fluxes, errors, pred_groups, unique_wavelengths, wavelength_to_color, bi, j, k, max_lc):
    plt.figure(figsize=(4, 4))
    ax = plt.gca()

    title = 'Flux vs. Time with Error Bars by Wavelength (Expert Alignment: {:.2f})'.format(max_lc)
    #ax.set_title(title, fontsize=10)
    
    for wavelength in unique_wavelengths:
        indices = [i for i, w in enumerate(wavelengths) if w == wavelength]
        ax.errorbar([times[i] for i in indices], [fluxes[i] for i in indices], yerr=[errors[i] for i in indices],
                    fmt='o', color=wavelength_to_color[wavelength], capsize=5, label=f'{int(wavelength)}')
    
    for i in range(len(pred_groups[j][k])):
        if pred_groups[j][k][i] == 1:
            if i == 0 or pred_groups[j][k][i-1] == 0:  # Start of a new span
                start_time = times[i]
            if i == len(pred_groups[j][k]) - 1 or pred_groups[j][k][i+1] == 0:  # End of a span
                end_time = times[i]
                ax.axvspan(start_time, end_time, color='gray', alpha=0.3)  # Adding the vertical span
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Flux')
    ax.legend(title="Wavelengths", loc='upper right', fontsize='small', title_fontsize='small')
    ax.grid(False)  # Disable grid explicitly if needed
    plt.savefig(f'groups_example/plot_{bi}_{j}_{k}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_linear_regression_vectorized(x, y, min_len=3):
    if x.numel() < min_len:
        return (1000, 1000)
    x = x.type(torch.float64)
    y = y.type(torch.float64)
    X = torch.vstack([x, torch.ones_like(x)]).T
    solution = torch.linalg.lstsq(X, y).solution
    slope = solution[0].item()
    intercept = solution[1].item()
    return slope, intercept

def threshold_f_vectorized(eps, error, time):
    if error is None or time is None or error.numel() == 0 or time.numel() == 0:
        return 0.0
    time_range = time.max() - time.min()
    if time_range.item() == 0:
        return 0.0
    thres = eps * torch.mean(error) / time_range
    return thres.item()

def get_fix_scores(test_dataloader, baseline, device, eps=1, min_len=3, window_size=50, step_size=25):
    fix_list = []
    with torch.no_grad():
        for bi, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items() if k != "objid"}
            times_wv_column = batch['past_time_features']
            target_column = batch['past_values']
            x_column = torch.cat((times_wv_column, target_column), dim=2)
            time_values, wavelength_values, flux_values, flux_error_values = x_column[:, :, 0], x_column[:, :, 1], x_column[:, :, 2], x_column[:, :, 3]

            diff_time = torch.diff(time_values, dim=1)
            non_increasing = diff_time <= 0
            padded_non_increasing = torch.nn.functional.pad(non_increasing, (1, 0), "constant", False)
            int_non_increasing = padded_non_increasing.int()
            valid_lengths = torch.argmax(int_non_increasing, dim=1)
            valid_lengths = torch.where(valid_lengths == 0, time_values.size(1), valid_lengths)

            valid_time_values_batch, valid_wavelength_values_batch, valid_flux_values_batch, valid_flux_error_values_batch = zip(*[
                (time_values[i, :l].clone().detach(), wavelength_values[i, :l].clone().detach(), flux_values[i, :l].clone().detach(), flux_error_values[i, :l].clone().detach()) for i, l in enumerate(valid_lengths)])

            pred_groups_batch = [baseline(l.item()) for l in valid_lengths]

            for j in range(len(valid_time_values_batch)):
                times, fluxes, errors, wavelengths = map(lambda x: x[j], [valid_time_values_batch, valid_flux_values_batch, valid_flux_error_values_batch, valid_wavelength_values_batch])
                unique_wavelengths = sorted(set(wavelengths.cpu().numpy()))
                num_unique_wavelengths = len(unique_wavelengths)

                time_wv = {w: [] for w in unique_wavelengths}
                flux_wv = {w: [] for w in unique_wavelengths}
                error_wv = {w: [] for w in unique_wavelengths}

                for time, flux, error, wavelength in zip(times.tolist(), fluxes.tolist(), errors.tolist(), wavelengths.tolist()):
                    time_wv[wavelength].append(time)
                    flux_wv[wavelength].append(flux)
                    error_wv[wavelength].append(error)

                color_map = plt.get_cmap('rainbow')
                colors = color_map(np.linspace(0, 1, len(unique_wavelengths)))
                wavelength_to_color = {w: c for w, c in zip(unique_wavelengths, colors)}
                
                alignment_scores_all = []
                for k in range(len(pred_groups_batch[j])):
                    group_time = torch.tensor([times[i] for i, value in enumerate(pred_groups_batch[j][k]) if value == 1])
                    group_slope = []
                    group_intercept = []
                    group_threshold = []
                    group_p = []
                    group_f = []

                    chunk_time = torch.arange(group_time[0], group_time[-1], step_size).unsqueeze(1)
                    chunk_time = torch.cat([chunk_time, chunk_time + window_size], dim=1)
                    chunk_time = torch.clamp(chunk_time, max = group_time[-1])
                    
                    if len(chunk_time) > 0:
                        for m in range(num_unique_wavelengths):
                            wavelength = unique_wavelengths[m]
                            crop_time_wv = torch.tensor(time_wv[wavelength])
                            mask = torch.isin(crop_time_wv, group_time)
                            crop_time_wv = crop_time_wv[mask]
                            crop_flux_wv = torch.tensor(flux_wv[wavelength])[mask]
                            crop_error_wv = torch.tensor(error_wv[wavelength])[mask]
                            
                            slope, intercept = perform_linear_regression_vectorized(crop_time_wv.float(), crop_flux_wv.float())
                            group_slope.append(slope)
                            group_intercept.append(intercept)
                            threshold = threshold_f_vectorized(eps, crop_error_wv.float(), crop_time_wv.float())
                            group_threshold.append(threshold)
    
                            predicted_flux = slope * crop_time_wv + intercept
                            correct_preds = (predicted_flux >= crop_flux_wv - (eps * crop_error_wv)) & (predicted_flux <= crop_flux_wv + (eps * crop_error_wv))
                            perc = correct_preds.sum().item() / len(crop_time_wv) if len(crop_time_wv) > 0 else 0
                            group_f.append(perc)
                            
                            p_in = 0
                            for start, end in chunk_time:
                                chunk_mask = (crop_time_wv >= start) & (crop_time_wv <= end)
                                if torch.any(chunk_mask):
                                    p_in += 1
                            group_p.append(p_in / len(chunk_time))
                        
                        group_lc = [p * f for p, f in zip(group_p, group_f)]
                        if all(slope <= threshold for slope, threshold in zip(group_slope, group_threshold)):
                            max_lc = max(group_lc)
                        else:
                            max_lc = max(lc for slope, lc, threshold in zip(group_slope, group_lc, group_threshold) if slope > threshold)
                        alignment_scores_all.append(max_lc)
                        
                pred_groups_batch_j = torch.tensor(pred_groups_batch[j])
                alignment_scores_all = torch.tensor(alignment_scores_all)
                group_alignment = torch.zeros(pred_groups_batch_j.shape[1])

                if alignment_scores_all.shape[0] == pred_groups_batch_j.shape[0]:
                    for idx in range(pred_groups_batch_j.shape[1]):
                        mask = pred_groups_batch_j[:, idx].bool()
                        if mask.any() and mask.shape[0] == alignment_scores_all.shape[0]:
                            selected_scores = alignment_scores_all[mask]
                            if selected_scores.numel() > 0:
                                group_alignment[idx] = selected_scores.mean() 
                fix_list.append(group_alignment.mean().item())
            fix_tensor = torch.tensor(fix_list)
            
    return fix_tensor.float().mean().item()
