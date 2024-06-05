import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage
sys.path.append("../src")
import exlib
from exlib.datasets.cholecystectomy import CholecDataset, CholecModel, CholecMetric

class GridHighLevelFeatures(nn.Module):
    # Let's assume image is 360x640 and make 40x40 grids (i.e., 9x16 partitions)
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, _, _, _ = x.shape
        mask_small = torch.tensor(range(9*16)).view(1,1,9,16).repeat(N,1,1,1)
        mask_big = F.interpolate(mask_small.float(), scale_factor=40).round().long()
        return mask_big.view(N,360,640)


class QuickShiftFeatures(nn.Module):
    # Use quickshift to perform image segmentation
    def __init__(self, kernel_size=10, max_dist=20, sigma=5, max_segs=40):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.sigma = sigma
        self.max_segs = max_segs

    def quickshift(self, image):
        # image is (C,H,W)
        image_np = image.numpy().transpose(1,2,0)
        segs = skimage.segmentation.quickshift(image_np, kernel_size=self.kernel_size, max_dist=self.max_dist, sigma=self.sigma)
        segs = torch.tensor(segs)
        segs[segs >= self.max_segs] = self.max_segs - 1
        return segs.long() # (H,W) of integers

    def forward(self, x):
        # x: (N,C,H,W)
        segs = torch.stack([self.quickshift(xi.cpu()) for xi in x]) # (N,H,W)
        return segs.to(x.device)




def get_cholec_scores(baselines = ['patch', 'quickshift']):
    dataset = CholecDataset(split="all_data")
    gonogo_model = CholecModel.from_pretrained("BrachioLab/cholecystectomy_gonogo").eval()
    
    metric = CholecMetric()
    torch.manual_seed(1234)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    all_baselines_scores = {}
    for baseline in baselines:
        if baseline == 'patch': # gridding 
            extractor = GridHighLevelFeatures()
        elif baseline == 'quickshift': # quickshift
            extractor = QuickShiftFeatures()

        scores = []
        for i, item in enumerate(tqdm(dataloader)):
            image = item["image"]
            with torch.no_grad():
                organs_masks = F.one_hot(item["organs"]).permute(0,3,1,2)
                masks = F.one_hot(extractor(image)).permute(0,3,1,2)
                
                score = metric(masks, organs_masks) # (N,H,W)
                scores.append(score.mean(dim=(1,2)))
            if i > 5:
                break
                
        scores = torch.cat(scores)
        print(f"Avg alignment of {baseline} features: {scores.mean():.4f}")
        all_baselines_scores[baseline] = scores
    
    return all_baselines_scores
    