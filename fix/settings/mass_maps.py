import sys
sys.path.append('../src')
import exlib
import torch
from datasets import load_dataset
from exlib.datasets import massmaps
from exlib.datasets.massmaps import MassMapsConvnetForImageRegression

# Baseline
from skimage.segmentation import watershed, quickshift
from scipy import ndimage
from skimage.feature import peak_local_max
import sys
sys.path.append('../src')
from exlib.explainers.common import convert_idx_masks_to_bool, patch_segmenter
import torch
import torch.nn as nn
import numpy as np
import cv2

from collections import defaultdict
import torch.nn.functional as F
from tqdm.auto import tqdm

# Alignment
from exlib.datasets.massmaps import MassMapsAlignment


class MassMapsPatch(nn.Module):
    def __init__(self, sz=(8, 8)):
        """
        sz : int, number of patches per side.
        """
        super().__init__()
        self.sz = sz
    
    def apply_patch(self, image):
        return patch_segmenter(image, sz=self.sz)
    
    def forward(self, images):
        """
        input: images (N, C=1, H, W)
        output: daf_preds (N, M, H, W)
        """
        daf_preds = []
        for image in images:
            segment_mask = torch.tensor(self.apply_patch(image[0].cpu().numpy())).to(images.device)
            masks_bool = convert_idx_masks_to_bool(segment_mask[None])
            daf_preds.append(masks_bool)
        daf_preds = torch.nn.utils.rnn.pad_sequence(daf_preds, batch_first=True)
        return daf_preds
        

class MassMapsQuickshift(nn.Module):
    def __init__(self, ratio=1.0, kernel_size=5, max_dist=10):
        """
        ratio : float, optional, between 0 and 1
            Balances color-space proximity and image-space proximity.
            Higher values give more weight to color-space.
        kernel_size : float, optional
            Width of Gaussian kernel used in smoothing the
            sample density. Higher means fewer clusters.
        max_dist : float, optional
            Cut-off point for data distances.
            Higher means fewer clusters.
        """
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        
    def apply_quickshift(self, image):
        ratio = self.ratio
        kernel_size = self.kernel_size
        max_dist = self.max_dist
        
        image = (image * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        raw_labels = quickshift(image_bgr, ratio=ratio, 
                                kernel_size=kernel_size, 
                                max_dist=max_dist)
        return raw_labels
    
    def forward(self, images):
        """
        input: images (N, C=1, H, W)
        output: daf_preds (N, M, H, W)
        """
        daf_preds = []
        for image in images:
            segment_mask = torch.tensor(self.apply_quickshift(image[0].cpu().numpy())).to(images.device)
            masks_bool = convert_idx_masks_to_bool(segment_mask[None])
            daf_preds.append(masks_bool)
        daf_preds = torch.nn.utils.rnn.pad_sequence(daf_preds, batch_first=True)
        return daf_preds


class MassMapsWatershed(nn.Module):
    def __init__(self, compactness=0, normalize=False):
        """
        compactness: Higher values result in more regularly-shaped watershed basins.
        """
        super().__init__()
        self.compactness = compactness
        self.normalize = normalize
        
    def apply_watershed(self, image):
        compactness = self.compactness
        normalize = self.normalize
        
        if normalize:
            # print('min', image.min(), 'max', image.max())
            image = (image - image.min()) / (image.max() - image.min())
            # print('after: min', image.min(), 'max', image.max())
        
        image = (image * 255).astype(np.uint8)
        distance = ndimage.distance_transform_edt(image)
        coords = peak_local_max(distance, min_distance=10, labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        raw_labels = watershed(-distance, markers, mask=image,
                               compactness=compactness)
        return raw_labels
    
    def forward(self, images):
        """
        input: images (N, C=1, H, W)
        output: daf_preds (N, M, H, W)
        """
        daf_preds = []
        for image in images:
            segment_mask = torch.tensor(self.apply_watershed(image[0].cpu().numpy())).to(images.device)
            masks_bool = convert_idx_masks_to_bool(segment_mask[None])
            daf_preds.append(masks_bool)
        daf_preds = torch.nn.utils.rnn.pad_sequence(daf_preds, batch_first=True)
        return daf_preds


def get_mass_maps_scores(baselines = ['patch', 'quickshift', 'watershed']): # currently we just assume we are running everything, need to update though to be able to specify a baseline to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load model
    model = MassMapsConvnetForImageRegression.from_pretrained(massmaps.MODEL_REPO) # BrachioLab/massmaps-conv
    model = model.to(device)
    
    # Load data
    train_dataset = load_dataset(massmaps.DATASET_REPO, split='train') # BrachioLab/massmaps-cosmogrid-100k
    val_dataset = load_dataset(massmaps.DATASET_REPO, split='validation')
    test_dataset = load_dataset(massmaps.DATASET_REPO, split='test')
    train_dataset.set_format('torch', columns=['input', 'label'])
    val_dataset.set_format('torch', columns=['input', 'label'])
    test_dataset.set_format('torch', columns=['input', 'label'])
    
    massmaps_align = MassMapsAlignment()
    
    # Eval
    watershed_baseline = MassMapsWatershed().to(device)
    watershed_norm_05_baseline = MassMapsWatershed(compactness=0.5, normalize=True).to(device)
    watershed_norm_1_baseline = MassMapsWatershed(compactness=1, normalize=True).to(device)
    quickshift_baseline = MassMapsQuickshift().to(device)
    patch_baseline = MassMapsPatch().to(device)
    
    baselines = {
        'watershed': watershed_baseline,
        'watershed_norm_05': watershed_norm_05_baseline,
        'watershed_norm_1': watershed_norm_1_baseline,
        'quickshift': quickshift_baseline,
        'patch': patch_baseline
    }
    
    batch_size = 5
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    
    model.eval()
    mse_loss_all = 0
    total = 0
    alignment_scores_all = defaultdict(list)
    
    with torch.no_grad():
        for bi, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            # if bi % 100 != 0:
            #     continue
            X = batch['input'].to(device)
            y = batch['label'].to(device)
            out = model(X)
            # loss
            loss = F.mse_loss(out, y, reduction='none')
            mse_loss_all = mse_loss_all + loss.sum(0)
            total += X.shape[0]
    
            # baseline
            for name, baseline in baselines.items():
                groups = baseline(X)
    
                # alignment
                alignment_scores = massmaps_align(groups, X)
                alignment_scores_all[name].extend(alignment_scores.flatten(1).cpu().numpy().tolist())
            
            
                
    loss_avg = mse_loss_all / total
    
    print(f'Omega_m loss {loss_avg[0].item():.4f}, sigma_8 loss {loss_avg[1].item():.4f}, avg loss {loss_avg.mean().item():.4f}')

    return alignment_scores_all