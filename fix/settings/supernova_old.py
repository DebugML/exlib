import pyarrow as pa
import pyarrow_hotfix
import torch
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import exlib
import math

from datasets import load_dataset
from collections import namedtuple
from exlib.datasets.pretrain import setup_model_config, get_dataset, get_dataset, setup_model_config
from exlib.datasets.dataset_preprocess_raw import create_train_dataloader_raw, create_test_dataloader_raw, create_test_dataloader
from exlib.datasets.informer_models import InformerConfig, InformerForSequenceClassification
from tqdm.auto import tqdm
pa.PyExtensionType.set_auto_load(True)
pyarrow_hotfix.uninstall()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# baseline
chunk_size = 30
def baseline(valid_length):
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


def perform_linear_regression(x, y, min_len=3):
    if not x or len(x) < min_len:
        return (1000, 1000)
    x = np.array(x)
    y = np.array(y)
    slope, intercept = np.polyfit(x, y, 1)
    return (slope, intercept)


def threshold_f(eps, error, time):
    if not error or not time:
        return 0
    time_range = max(time) - min(time)
    if time_range == 0:
        return 0
    thres = eps * (sum(error)/len(error)) / (time_range)
    return thres


def calculate_alignment_scores_for_group(times, unique_wavelengths, fluxes, errors, groups, time_wv, flux_wv, error_wv,  eps=1, min_len=3, window_size=50, step_size=25):
    group_time = [times[i] for i, value in enumerate(groups) if value == 1]
    group_slope = []
    group_intercept = []
    group_threshold = []
    group_lc = []
    group_p = []
    group_f = []
    #chunk_time = [[time, time + window_size] for time in crop_time_wv]
    chunk_time = []
    current_start = group_time[0]
    while current_start + window_size < group_time[-1]:
        chunk_time.append([current_start, current_start + window_size])
        current_start += step_size
    chunk_time.append([current_start, current_start + window_size])
    
    for m in range(len(unique_wavelengths)):
        chunk_f = 0
        chunk_p = 0
        crop_time_wv = [time for time in time_wv[unique_wavelengths[m]] if time in group_time]
        
        crop_flux_wv = [flux_wv[unique_wavelengths[m]][i] for i, time in enumerate(time_wv[unique_wavelengths[m]]) if time in group_time]
        crop_error_wv = [error_wv[unique_wavelengths[m]][i] for i, time in enumerate(time_wv[unique_wavelengths[m]]) if time in group_time]
        slope, intercept = perform_linear_regression(crop_time_wv, crop_flux_wv)
        group_slope.append(slope)
        group_intercept.append(intercept)
        threshold = threshold_f(eps, crop_error_wv, crop_time_wv)
        group_threshold.append(threshold)
        p_in = 0
        f_in = 0
        for n in range(len(chunk_time)):
            chunk_time_wv = [time for time in crop_time_wv if chunk_time[n][0] <= time <= chunk_time[n][1]]
            indices = [i for i, time in enumerate(crop_time_wv) if time in chunk_time_wv]
            chunk_flux_wv = [crop_flux_wv[i] for i in indices]
            chunk_error_wv = [crop_error_wv[i] for i in indices]
            chunk_slope = perform_linear_regression(chunk_time_wv, chunk_flux_wv)
            chunk_threshold = threshold_f(eps, chunk_error_wv, chunk_time_wv)
            if chunk_time_wv:
                p_in += 1
                
            f_in_chunk = 0
            if len(chunk_time_wv) >= min_len:
                for time, flux, error in zip(chunk_time_wv, chunk_flux_wv, chunk_error_wv):
                    predicted_flux = group_slope[m] * time + group_intercept[m]
                    lower_bound = flux - (eps*error)
                    upper_bound = flux + (eps*error)
                    if lower_bound <= predicted_flux <= upper_bound:
                        f_in_chunk += 1
            if f_in_chunk == len(chunk_time_wv):
                f_in += 1
            
        group_f.append(f_in / len(chunk_time))
        group_p.append(p_in / len(chunk_time))
    group_lc = []
    for p, f in zip(group_p, group_f):
        group_lc.append(p * f)

    if all(slope <= threshold for slope, threshold in zip(group_slope, group_threshold)):
        max_lc = max(group_lc)
        idx = torch.Tensor(group_lc).argmax()
        #print("wavelength 1", unique_wavelengths[idx])
    else:
        max_lc = max(lc for slope, lc, threshold in zip(group_slope, group_lc, group_threshold) if slope > threshold)
        
        idx = torch.Tensor(group_lc).argmax()
        #print("wavelength 2", unique_wavelengths[idx])
    return(max_lc)



def get_supernova_scores(baselines = ['chunk']):
    # load dataset
    dataset = load_dataset("BrachioLab/supernova-timeseries")
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']

    # load model
    model = InformerForSequenceClassification.from_pretrained("BrachioLab/supernova-classification")
    model = model.to(device)

    config = InformerConfig.from_pretrained("BrachioLab/supernova-classification")
    test_dataloader = create_test_dataloader_raw(
        config=config,
        dataset=test_dataset,
        batch_size=25,
        compute_loss=True
    )


    # alignment score - without ground truth
    with torch.no_grad():
        alignment_scores_all = []
        for bi, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            # prediction
            batch = {k: v.to(device) for k, v in batch.items() if k != "objid"}
                
            times_wv_column = batch['past_time_features'].to('cpu')
            target_column = batch['past_values'].to('cpu')
            x_column = np.concatenate((times_wv_column, target_column), axis=2) # time, wavelength, flux, flux_error
            time_values = x_column[:, :, 0].tolist() # time_values is from 0 to 1, and if it is less than 300 random values
            wavelength_values = x_column[:, :, 1].tolist()
            flux_values = x_column[:, :, 2].tolist()
            flux_error_values = x_column[:, :, 3].tolist()
            
            # predicted group
            valid_time_values_batch = []
            valid_wavelength_values_batch = []
            valid_flux_values_batch = []
            valid_flux_error_values_batch = []
            valid_length_batch = []
            pred_groups_batch = []
            
            for idx, time_list in enumerate(time_values):
                valid_length = next((j for j in range(1, len(time_list)) if time_list[j] <= time_list[j-1]), len(time_list))
                valid_time_values_batch.append(time_list[:valid_length])
                valid_wavelength_values_batch.append(wavelength_values[idx][:valid_length])
                valid_flux_values_batch.append(flux_values[idx][:valid_length])
                valid_flux_error_values_batch.append(flux_error_values[idx][:valid_length])
                valid_length_batch.append(valid_length)
                
                pred_groups = baseline(valid_length)
                pred_groups_batch.append(pred_groups)
                # pred_groups_batch: batch_size * pred_group_num * valid_length

            for j in range(len(valid_time_values_batch)): # j: batch size
                times = valid_time_values_batch[j]
                fluxes = valid_flux_values_batch[j]
                errors = valid_flux_error_values_batch[j]
                wavelengths = valid_wavelength_values_batch[j]
                unique_wavelengths = sorted(set(wavelengths))
                num_unique_wavelengths = len(unique_wavelengths)
                time_wv = {wavelength: [] for wavelength in unique_wavelengths}
                flux_wv = {wavelength: [] for wavelength in unique_wavelengths}
                error_wv = {wavelength: [] for wavelength in unique_wavelengths}
                
                for time, flux, error, wavelength in zip(times, fluxes, errors, wavelengths):
                    time_wv[wavelength].append(time)
                    flux_wv[wavelength].append(flux)
                    error_wv[wavelength].append(error)
                unique_wavelengths = sorted(set(wavelengths))
                color_map = plt.get_cmap('rainbow')
                colors = color_map(np.linspace(0, 1, len(unique_wavelengths)))
                wavelength_to_color = {w: c for w, c in zip(unique_wavelengths, colors)}

                eps = 1
                min_len = 3
                window_size = 50
                step_size = int(window_size / 2)
                for k in range(len(pred_groups_batch[j])): # k: number of group
                    group_time = [times[i] for i, value in enumerate(pred_groups_batch[j][k]) if value == 1]
                    group_slope = []
                    group_intercept = []
                    group_threshold = []
                    group_lc = []
                    group_p = []
                    group_f = []
                    #chunk_time = [[time, time + window_size] for time in crop_time_wv]
                    chunk_time = []
                    current_start = group_time[0]
                    while current_start + window_size < group_time[-1]:
                        chunk_time.append([current_start, current_start + window_size])
                        current_start += step_size
                    chunk_time.append([current_start, current_start + window_size])
                    for m in range(len(unique_wavelengths)):
                        chunk_f = 0
                        chunk_p = 0
                        crop_time_wv = [time for time in time_wv[unique_wavelengths[m]] if time in group_time]
                        
                        crop_flux_wv = [flux_wv[unique_wavelengths[m]][i] for i, time in enumerate(time_wv[unique_wavelengths[m]]) if time in group_time]
                        crop_error_wv = [error_wv[unique_wavelengths[m]][i] for i, time in enumerate(time_wv[unique_wavelengths[m]]) if time in group_time]
                        slope, intercept = perform_linear_regression(crop_time_wv, crop_flux_wv)
                        group_slope.append(slope)
                        group_intercept.append(intercept)
                        threshold = threshold_f(eps, crop_error_wv, crop_time_wv)
                        group_threshold.append(threshold)
                        p_in = 0
                        pos_f = 0
                        if len(crop_time_wv) >= min_len:
                            for time, flux, error in zip(crop_time_wv, crop_flux_wv, crop_error_wv):
                                predicted_flux = group_slope[m] * time + group_intercept[m]
                                lower_bound = flux - (eps*error)
                                upper_bound = flux + (eps*error)
                                if lower_bound <= predicted_flux <= upper_bound:
                                    pos_f += 1
                        if len(crop_time_wv) == 0:
                            perc = 0
                        else:
                            perc = pos_f / len(crop_time_wv)
                        group_f.append(perc)
                        for n in range(len(chunk_time)):
                            chunk_time_wv = [time for time in crop_time_wv if chunk_time[n][0] <= time <= chunk_time[n][1]]
                            indices = [i for i, time in enumerate(crop_time_wv) if time in chunk_time_wv]
                            chunk_flux_wv = [crop_flux_wv[i] for i in indices]
                            chunk_error_wv = [crop_error_wv[i] for i in indices]
                            if chunk_time_wv:
                                p_in += 1
                        group_p.append(p_in / len(chunk_time))
                    group_lc = []
                    for p, f in zip(group_p, group_f):
                        group_lc.append(p * f)
                    if all(slope <= threshold for slope, threshold in zip(group_slope, group_threshold)):
                        max_lc = max(group_lc)
                    else:
                        max_lc = max(lc for slope, lc, threshold in zip(group_slope, group_lc, group_threshold) if slope > threshold)
                    alignment_scores_all.append(max_lc)    
    print("score:", sum(alignment_scores_all)/len(alignment_scores_all), len(alignment_scores_all))
    return alignment_scores_all