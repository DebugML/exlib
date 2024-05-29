import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob
from collections import namedtuple
from typing import Optional, Union, List

import segmentation_models_pytorch as smp
from ..modules.sop import SOPImageSeg, SOPConfig, get_chained_attr


# The kinds of splits we can do
SPLIT_TYPES = ["train_frame", "test_frame", "train_video", "test_video"]

# Splitting images by video source
VIDEO_GLOBS = \
      [f"AdnanSet_LC_{i}_*" for i in range(1,165)] \
    + [f"AminSet_LC_{i}_*" for i in range(1,11)] \
    + [f"cholec80_video0{i}_*" for i in range(1,10)] \
    + [f"cholec80_video{i}_*" for i in range(10,81)] \
    + ["HokkaidoSet_LC_1_*", "HokkaidoSet_LC_2_*"] \
    + [f"M2CCAI2016_video{i}_*" for i in range(81,122)] \
    + [f"UTSWSet_Case_{i}_*" for i in range(1,13)] \
    + [f"WashUSet_LC_01_*"]

#
class AbdomenOrgans(Dataset):
    def __init__(
        self,
         data_dir: str,
         images_dir: str = "images",
         gonogo_labels_dir: str = "gonogo_labels",
         organ_labels_dir: str = "organ_labels",
         split: str = "train_frame",
         train_ratio: float = 0.8,
         image_height: float = 384,
         image_width: float = 640,
         train_angle_max: float = 60.0,
         image_transforms = None,
         label_transforms = None,
         download: bool = False,
         gen_seed: int = 1234
    ):
        if download:
            raise ValueError("download not implemented")

        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, images_dir)
        self.gonogo_labels_dir = os.path.join(data_dir, gonogo_labels_dir)
        self.organ_labels_dir = os.path.join(data_dir, organ_labels_dir)

        assert os.path.isdir(self.images_dir)
        assert os.path.isdir(self.gonogo_labels_dir)
        assert os.path.isdir(self.organ_labels_dir)
        assert split in SPLIT_TYPES
        self.split = split

        gen = torch.Generator()
        gen.manual_seed(gen_seed)

        # Split not regarding video
        if split == "train_frame" or split == "test_frame":
            all_image_files = sorted(os.listdir(self.images_dir))
            num_all, num_train = len(all_image_files), int(len(all_image_files) * train_ratio)
            perm = torch.randperm(num_all, generator=gen)
            idxs = perm[:num_train] if split.startswith("train") else perm[num_train:]
            self.image_files = sorted([all_image_files[i] for i in idxs])

        # Split by the video source
        elif split == "train_video" or split == "test_video":
            num_all, num_train = len(VIDEO_GLOBS), int(len(VIDEO_GLOBS) * train_ratio)
            perm = torch.randperm(num_all, generator=gen)
            idxs = perm[:num_train] if "train_frame" in split else perm[num_train:]

            image_files = []
            for i in idxs:
                image_files += glob.glob(os.path.join(self.images_dir, VIDEO_GLOBS[i]))
            self.image_files = sorted(image_files)

        else:
            raise NotImplementedError()

        # Random rotation angles used for the training data
        self.train_angle_max = train_angle_max
        self.random_angles = train_angle_max * (torch.rand(len(self.image_files)) * 2 - 1)

        self.image_height = image_height
        self.image_width = image_width

        # Image transforms
        if image_transforms is None:
            self.image_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_height, image_width), antialias=True),
            ])
        else:
            assert callable(image_transforms)
            self.image_transforms = image_transforms

        if label_transforms is None:
            self.label_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_height, image_width), antialias=True)
            ])
        else:
            assert callable(label_transforms)
            self.label_transforms = label_transforms


    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        image_file = os.path.join(self.images_dir, self.image_files[idx])
        organ_label_file = os.path.join(self.organ_labels_dir, self.image_files[idx])
        gonogo_label_file = os.path.join(self.gonogo_labels_dir, self.image_files[idx])

        # Read image and label
        image = Image.open(image_file).convert("RGB")
        organ_label = Image.open(organ_label_file).convert("L") # L is grayscale
        gonogo_label = Image.open(gonogo_label_file).convert("L")

        if self.split.startswith("train"):
            image = self.image_transforms(image)
            organ_label = self.label_transforms(organ_label)
            gonogo_label = self.label_transforms(gonogo_label)

            # Apply the random rotation
            angle = self.random_angles[idx].item()
            image = transforms.functional.rotate(image, angle)
            organ_label = transforms.functional.rotate(organ_label, angle)
            gonogo_label = transforms.functional.rotate(gonogo_label, angle)

        else:
            image = self.image_transforms(image)
            organ_label = self.label_transforms(organ_label)
            gonogo_label = self.label_transforms(gonogo_label)

        organ_label = (organ_label * 255).round().long()
        gonogo_label = (gonogo_label * 255).round().long()
        return image, organ_label, gonogo_label



# Basic classification model
class AbdomenClsModel(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1),   # (N,256,32,32)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1), # (N,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  # (N,64,8,8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),   # (N,32,4,4)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(32*4*4, num_classes)
        )

    def forward(self, x):
        N, C, H, W = x.shape
        assert C == 3
        x = F.interpolate(x, size=[64,64])
        y = self.layers(x)
        return y


# Basic segmentation model
class AbdomenSegModel(nn.Module):
    def __init__(self, num_segments, in_channels=3,
                 encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.unet = smp.Unet(encoder_name=encoder_name,
                             encoder_weights=encoder_weights,
                             in_channels=in_channels,
                             classes=num_segments)

    def forward(self, x):
        N, C, H, W = x.shape
        assert H % 32 == 0 and W % 32 == 0
        return self.unet(x)


ModelOutput = namedtuple("ModelOutput", ["logits", "pooler_output"])

def convert_idx_masks_to_bool(masks, num_masks=None):
    """
    input: masks (1, img_dim1, img_dim2)
    output: masks_bool (num_masks, img_dim1, img_dim2)
    """
    if num_masks is not None:
        unique_idxs = torch.arange(num_masks).to(masks.device)
    else:
        unique_idxs = torch.sort(torch.unique(masks)).values
    idxs = unique_idxs.view(-1, 1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[-2], 
                                     masks.shape[-1])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool


class Unet(smp.Unet):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__(encoder_name=encoder_name,
                         encoder_depth=encoder_depth,
                         encoder_weights=encoder_weights,
                         decoder_use_batchnorm=decoder_use_batchnorm,
                         decoder_channels=decoder_channels,
                         decoder_attention_type=decoder_attention_type,
                         in_channels=in_channels,
                         classes=classes,
                         activation=activation,
                         aux_params=aux_params)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return ModelOutput(logits=masks,
                           pooler_output=decoder_output)


class OrganGNGModel(nn.Module):
    def __init__(self, num_organs=4):
        super(OrganGNGModel, self).__init__()
        self.num_organs = num_organs
        self.organ_seg_model = None
        self.gonogo_model = None
    
    def restore_organ(self, restore_organ=None):
        if restore_organ:
            state_dict = torch.load(restore_organ)
            self.organ_seg_model.load_state_dict(state_dict['model_state_dict'], 
                                                strict=False)

    def restore_gonogo(self, restore_gonogo=None):
        if restore_gonogo:
            state_dict = torch.load(restore_gonogo)
            self.gonogo_model.load_state_dict(state_dict['model_state_dict'], 
                                              strict=False)


class OrganGNGSopSelectorModel(OrganGNGModel):
    def __init__(self, num_organs=4, wrapper_config_filepath='organs/abdomen_unet_wrapper_a.json'):
        super().__init__(num_organs=num_organs)
        self.organ_seg_model = Unet(
                encoder_name = "resnet50",
                encoder_weights = "imagenet",
                in_channels = 3,
                classes = num_organs,
                activation = "softmax2d")

        self.gonogo_model = Unet(
                encoder_name = "resnet50",
                encoder_weights = "imagenet",
                in_channels = 3,
                classes = 3,
                activation = "identity")
        
        self.wrapper_config_filepath = wrapper_config_filepath
        print('wrapper_config_filepath', wrapper_config_filepath)
        config = SOPConfig(json_file=wrapper_config_filepath)

        # allow specifying args to be different from in the json file
        # print(get_chained_attr(self.gonogo_model, config.finetune_layers[0])[0].weight.shape)
        class_weights = get_chained_attr(self.gonogo_model, config.finetune_layers[0])[0].weight.view(3,-1).clone()
        self.sop_model = SOPImageSeg(config, 
                              self.gonogo_model,
                              class_weights =class_weights
                                    )
    
    def freeze_organ(self):
        for name, param in self.organ_seg_model.named_parameters():
            param.requires_grad = False

    def forward(self, x, get_organs=False, return_dict=False):
        organ_output = self.organ_seg_model(x)
        organ_mask_output = organ_output.logits.argmax(dim=1).byte()
        organ_mask_output_bools = []
        for organ_mask_output_i in organ_mask_output:
            organ_mask_output_bool = convert_idx_masks_to_bool(organ_mask_output_i, 
                                                               num_masks=self.num_organs).int()
            organ_mask_output_bools.append(organ_mask_output_bool)
        organ_mask_output_bools = torch.stack(organ_mask_output_bools)
        gonogo_output = self.sop_model(x, input_mask_weights=organ_mask_output_bools, epoch=0,
                                       mask_batch_size=8, return_tuple=True)
        
        if return_dict:
            return {
                'organ_output': organ_output,
                'gonogo_output': gonogo_output
            }

        if get_organs:
            return organ_output.logits, gonogo_output.logits
        return gonogo_output.logits
    
    
def miou(predict, label, class_num=3):
    """
        get miou of multiple batches
        @param predict: Shape:[B, W, H]
        @param label: Shape:[B, W, H]
    """
    batch = label.shape[0]
    predict, label = predict.flatten(), label.flatten()  
    #select pixels to be segmentated
    k = (predict >= 0) & (predict < class_num) 
    hist = torch.bincount(class_num * predict[k].type(torch.int32) + label[k], minlength=batch * (class_num ** 2)).reshape(batch, class_num, class_num)
    hist = hist.sum(0)

    miou = torch.diag(hist) / torch.maximum((hist.sum(1) + hist.sum(0) - torch.diag(hist)), torch.tensor(1))
    return miou.mean()


def dice_acc(predict, label):
    """
        get dice coefficient of multiple batches
        @param predict: Shape:[B, C, W, H]  C = num_classes
        @param label: Shape:[B, C, W, H]
    """
    # import pdb; pdb.set_trace()
    batch_num, class_num = predict.shape[0: 2]
    assert predict.size() == label.size(), "the size of predict and target must be equal."
    
    intersection = (predict * label).reshape((batch_num, class_num, -1)).sum(dim=2)
    union = (predict + label).reshape((batch_num, class_num, -1)).sum(dim=2)
    dice = (2. * intersection + 1) / (union + 1)
    dice = dice.mean()
    return dice


def mpa(predict, label, class_num=3):
    """
       get MPA of multiple batches
        @param predict: Shape:[B, W, H]
        @param label: Shape:[B, W, H]
    """
    batch = label.shape[0]
    predict, label = predict.flatten(), label.flatten() 
    k = (predict >= 0) & (predict < class_num) 
    hist = torch.bincount(class_num * predict[k].type(torch.int32) + label[k], minlength=batch * (class_num ** 2)).reshape(batch, class_num, class_num)
    hist = hist.sum(0)
    acc_cls = torch.diag(hist) / hist.sum(axis=1)
    acc_cls = torch.nanmean(acc_cls)    #.type(torch.float32)
    return acc_cls



def weighted_dice_loss(predict, label):
    """
    Computes the weighted dice loss for multi-class image segmentation.
    Args:
        predict: Tensor of predicted values with shape (batch_size, num_classes, height, width)
        label: Tensor of ground truth labels with shape (batch_size, num_classes, height, width)
    Returns:
        Weighted dice loss tensor
    """
    batch_size = predict.size(0)
    num_classes = predict.size(1)
    smooth = 1e-5  # smoothing factor to avoid division by zero
    total_loss = 0.0
    
    for i in range(batch_size):
        for j in range(num_classes):
            predict_mask = predict[i, j, :, :]
            label_mask = label[i, j, :, :]
            intersection = (predict_mask * label_mask).sum()
            area_predict = predict_mask.sum()  # sum of all possibilities
            area_label = label_mask.sum()  #sum of all ones 

            weight = area_label/(640*360 + smooth)
            loss = 1 - ((2.0 * intersection + smooth) / (area_predict + area_label + smooth)) * weight
            total_loss +=  loss
            
    return total_loss / (batch_size * num_classes)
