import torch
import gzip
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.io
import pickle
from torch import nn
from ..modules.sop import get_mask_transform


class CosmogridDataset(Dataset):
    def __init__(self, data_dir, split='train', data_size=-1, mask_path=None,
                 inputs_filename='X_maps_Cosmogrid_100k.npy',
                 labels_filename='y_maps_Cosmogrid_100k.npy',
                 mask_transform=None,
                 num_masks_max=200,
                 download=False,
                 mask_big_first=False,
                 even_data_sample=False):
        if download: 
            raise ValueError("download not implemented")
        
        self.split = split
        self.data_dir = data_dir
        self.data_size = data_size
        self.mask_path = mask_path
        if mask_transform is None:
            mask_transform = get_mask_transform(num_masks_max=num_masks_max, 
                                                big_first=mask_big_first)
        self.mask_transform = mask_transform
        self.even_data_sample = even_data_sample
        # load cosmological parameters -------
        # Omega_m, H0, ns, sigma_8, w, omega_b
        Xvals = np.load(os.path.join(data_dir, inputs_filename), allow_pickle=True)
        Yvals = np.load(os.path.join(data_dir, labels_filename), allow_pickle=True)

        # number of samples
        num_samples = len(Yvals)

        # split the sample for training ----------
        train_split, val_split, test_split = int(0.80*num_samples), \
                    int(0.10*num_samples), int(0.10*num_samples)
            
        print('# samples used for training:', train_split)
        print('# samples used for validation:', val_split)
        print('# samples used for testing:' ,test_split)
        print('# total samples:', train_split+val_split+test_split)
        
        np.random.seed(42)
        train_x, val_x, test_x = np.split(Xvals, [train_split, train_split+val_split])
        train_y, val_y, test_y = np.split(Yvals, [train_split, train_split+val_split])
        print('x shape', train_x.shape, val_x.shape, test_x.shape)
        print('y shape', train_y.shape, val_y.shape, test_y.shape)
        if mask_path is not None:
            masks_vals = np.load(mask_path, allow_pickle=True)
            train_masks, val_masks, test_masks = np.split(masks_vals, 
                                                          [train_split, 
                                                           train_split+val_split])
            print('masks shape', train_masks.shape, val_masks.shape, test_masks.shape)
        else:
            train_masks, val_masks, test_masks = None, None, None

        params_mask = np.array([True,False,False,True,False,False])
        self.output_num = len(params_mask[params_mask])

        # let's focus on omega_m and sigma_8 
        train_y, val_y, test_y = train_y[:,params_mask], val_y[:,params_mask], test_y[:,params_mask]

        self.splits = {
            'train': (train_x, train_y, train_masks),
            'val': (val_x, val_y, val_masks),
            'test': (test_x, test_y, test_masks)
        }

        self.images = self.splits[split][0][:, None, :, :]
        self.labels = self.splits[split][1]
        self.masks = self.splits[split][2]
        print('-- ALL --')
        print('max', self.images.max())
        print('min', self.images.min())
        # print('mean', self.images.mean().item())
        # import pdb
        # pdb.set_trace()
        # print('mode', torch.tensor(self.images).view(-1).mode().values.item())

        if data_size != -1:
            if not even_data_sample:
                self.images = self.images[:data_size]
                self.labels = self.labels[:data_size]
                if self.masks is not None:
                    self.masks = self.masks[:data_size]
            else:
                # sample data with even interval
                data_interval = len(self.images) // data_size
                self.images = self.images[::data_interval]
                self.labels = self.labels[::data_interval]
                if self.masks is not None:
                    self.masks = self.masks[::data_interval]

        print(f'-- SPLIT {split} --')
        print('max', self.images.max().item())
        print('min', self.images.min().item())

        print(f'Finished loading {len(self.labels)} {split} images ... ')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.masks is not None:
            mask_i = self.masks[idx]
            mask = torch.tensor(mask_i)
            mask = self.mask_transform(mask)
            return image, label, mask, mask_i
        else:
            return image, label
        

class ModelOutput:
    def __init__(self, logits, pooler_output):
        self.logits = logits
        self.pooler_output = pooler_output
        
class CNNModel(nn.Module):
    def __init__(self, output_num):
        super(CNNModel, self).__init__()
        
        # self.normalization = nn.BatchNorm2d(1)
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
        self.fc4 = nn.Linear(32, output_num)
        
    def forward(self, x):
        # x = self.normalization(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
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
        return ModelOutput(logits=logits,
                           pooler_output=pooler_output)
    