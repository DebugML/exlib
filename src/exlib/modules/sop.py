# TODO: optimize

from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import PreTrainedModel
import collections.abc
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
# from transformers import PretrainedConfig, PreTrainedModel
import copy
import json
from collections import namedtuple
import os
import math

import collections
import torchvision

EPS = 1e-8

AttributionOutputSOP = namedtuple("AttributionOutputSOP", 
                                  ["logits",
                                   "logits_all",
                                   "pooler_outputs_all",
                                   "masks",
                                   "mask_weights",
                                   "attributions", 
                                   "attributions_max",
                                   "attributions_all",
                                   "flat_masks",
                                   "grouped_attributions"])


def gaussian_kernel(size, sigma, device):
    """Generates a 2D Gaussian kernel."""
    coords = torch.tensor([(x - size // 2) for x in range(size)]).to(device)
    grid = coords.unsqueeze(0).repeat(size, 1)
    kernel = torch.exp(-(grid ** 2 + grid.t() ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel

def gaussian_blur(image, kernel_size=5, sigma=1):
    """Applies Gaussian blur to an image."""
    channels = image.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, image.device)
    # Reshape to 2d depthwise convolutional weight
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = kernel_size // 2
    blurred_image = F.conv2d(image, kernel, padding=padding, groups=channels)
    return blurred_image
    
def convert_idx_masks_to_bool(masks):
    """
    input: masks (1, img_dim1, img_dim2)
    output: masks_bool (num_masks, img_dim1, img_dim2)
    """
    unique_idxs = torch.sort(torch.unique(masks)).values
    idxs = unique_idxs.view(-1, 1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[1], 
                                     masks.shape[2])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool


def convert_idx_masks_to_bool_big_first(masks):
    """
    input: masks (1, img_dim1, img_dim2)
    output: masks_bool (num_masks, img_dim1, img_dim2)
    """
    unique_idxs, counts = torch.unique(masks, return_counts=True)
    unique_idxs_sorted = unique_idxs[counts.argsort(descending=True)]
    idxs = unique_idxs_sorted.view(-1, 1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[1], 
                                     masks.shape[2])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool


def get_mask_transform(num_masks_max=200, processor=None, big_first=False):
    def mask_transform(mask):
        seg_mask_cut_off = num_masks_max
        # Preprocess the mask using the ViTImageProcessor
        if len(mask.shape) == 2 and mask.dtype == torch.bool:
            mask_dim1, mask_dim2 = mask.shape
            mask = mask.unsqueeze(0).expand(3, 
                                            mask_dim1, 
                                            mask_dim2).float()
            if processor is not None:
                inputs = processor(mask, 
                                do_rescale=False, 
                                do_normalize=False,
                                return_tensors='pt')
                # (1, 3, 224, 224)
                return inputs['pixel_values'][0][0]
            else:
                return mask
        else: # len(mask.shape) == 3
            if mask.dtype != torch.bool:
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)
                if big_first:
                    mask = convert_idx_masks_to_bool_big_first(mask)
                else:
                    mask = convert_idx_masks_to_bool(mask)
            bsz, mask_dim1, mask_dim2 = mask.shape
            mask = mask.unsqueeze(1).expand(bsz, 
                                            3, 
                                            mask_dim1, 
                                            mask_dim2).float()

            if bsz < seg_mask_cut_off:
                repeat_count = seg_mask_cut_off // bsz + 1
                mask = torch.cat([mask] * repeat_count, dim=0)

            # add additional mask afterwards
            mask_sum = torch.sum(mask[:seg_mask_cut_off - 1], dim=0, keepdim=True).bool()
            if False in mask_sum:
                mask = mask[:seg_mask_cut_off - 1]
                compensation_mask = (1 - mask_sum.int()).bool()
                mask = torch.cat([mask, compensation_mask])
            else:
                mask = mask[:seg_mask_cut_off]

            if processor is not None:
                inputs = processor(mask, 
                                do_rescale=False, 
                                do_normalize=False,
                                return_tensors='pt')
                
                return inputs['pixel_values'][:,0]
            else:
                return mask[:,0]
    return mask_transform


def get_chained_attr(obj, attr_chain):
    attrs = attr_chain.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def compress_single_masks(masks, masks_weights, min_size):
    # num_masks, seq_len = masks.shape
    masks_bool = (masks > 0).int()
    sorted_weights, sorted_indices = torch.sort(masks_weights, descending=True)
    sorted_indices = sorted_indices[sorted_weights > 0]

    
    try:
        masks_bool = masks_bool[sorted_indices]  # sorted masks
    except:
        import pdb; pdb.set_trace()
        masks_bool = masks_bool[sorted_indices]  # sorted masks
    
    masks = torch.zeros(*masks_bool.shape[1:]).to(masks.device)
    count = 1
    for mask in masks_bool:
        new_mask = mask.bool() ^ (mask.bool() & masks.bool())
        if torch.sum(new_mask) >= min_size:
            masks[new_mask] = count
            count += 1

    masks = masks - 1
    masks = masks.int()
    masks[masks == -1] = torch.max(masks) + 1

    return masks

def compress_masks(masks, masks_weights, min_size=0):
    new_masks = []
    for i in range(len(masks)):
        compressed_mask = compress_single_masks(masks[i], masks_weights[i], 
                                                min_size)
        new_masks.append(compressed_mask)
    return torch.stack(new_masks)


def compress_masks_image(masks, masks_weights, min_size=0):
    assert len(masks.shape) == 4 # bsz, num_masks, img_dim_1, img_dim_2 = masks.shape
    return compress_masks(masks, masks_weights, min_size)
    

def compress_masks_text(masks, masks_weights, min_size=0):
    assert len(masks.shape) == 3 # bsz, num_masks, seq_len = masks.shape
    return compress_masks(masks, masks_weights, min_size)
           

def _get_inverse_sqrt_with_separate_heads_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, 
    num_steps_per_epoch: int,
    timescale: int = None, 
    num_heads: int = 1, 
):
    epoch = current_step // (num_steps_per_epoch * num_heads)
    steps_within_epoch = current_step % num_steps_per_epoch
    step_for_curr_head = epoch * num_steps_per_epoch + steps_within_epoch
    if step_for_curr_head < num_warmup_steps:
        return float(step_for_curr_head) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((step_for_curr_head + shift) / timescale)
    return decay

def get_inverse_sqrt_with_separate_heads_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_steps_per_epoch: int,
    timescale: int = None, 
    num_heads: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if timescale is None:
        timescale = num_warmup_steps

    lr_lambda = partial(
        _get_inverse_sqrt_with_separate_heads_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_steps_per_epoch=num_steps_per_epoch,
        timescale=timescale,
        num_heads=num_heads,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


"""Sparsemax activation function.

Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
"""


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, inputs):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        device = inputs.device
        inputs = inputs.transpose(0, self.dim)
        original_size = inputs.size()
        inputs = inputs.reshape(inputs.size(0), -1)
        inputs = inputs.transpose(0, 1)
        dim = 1

        number_of_logits = inputs.size(dim)

        # Translate input by max for numerical stability
        inputs = inputs - torch.max(inputs, dim=dim, keepdim=True)[0].expand_as(inputs)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=inputs, dim=dim, descending=True)[0]
        range_tensor = torch.arange(start=1, end=number_of_logits + 1, step=1, 
                                    device=device, dtype=inputs.dtype).view(1, -1)
        range_tensor = range_tensor.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range_tensor * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(inputs.type())
        k = torch.max(is_gt * range_tensor, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(inputs)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(inputs), inputs - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


def same_tensor(tensor, *args):
    ''' Do the input tensors all point to the same underlying data '''
    for other in args:
        if not torch.is_tensor(other):
            return False

        if tensor.device != other.device:
            return False

        if tensor.dtype != other.dtype:
            return False

        if tensor.data_ptr() != other.data_ptr():
            return False

    return True


class SparseMultiHeadedAttention(nn.Module):
    ''' Implement a multi-headed attention module '''
    def __init__(self, embed_dim, num_heads=1, scale=1):
        ''' Initialize the attention module '''
        super(SparseMultiHeadedAttention, self).__init__()

        # ensure valid inputs
        assert embed_dim % num_heads == 0, \
            f'num_heads={num_heads} should evenly divide embed_dim={embed_dim}'

        # store off the scale and input params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.scale = scale
        # self.scale = self.projection_dim ** -0.5

        # Combine projections for multiple heads into a single linear layer for efficiency
        self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sparsemax = Sparsemax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialization '''
        # Initialize using Xavier
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.input_weights, gain)
        nn.init.xavier_uniform_(self.output_projection.weight, gain)

    def project(self, inputs, index=0, chunks=1):
        ''' Produce a linear projection using the weights '''
        batch_size = inputs.shape[0]
        start = index * self.embed_dim
        end = start + chunks * self.embed_dim
        # import pdb; pdb.set_trace()
        projections = F.linear(inputs, self.input_weights[start:end]).chunk(chunks, dim=-1)

        output_projections = []
        for projection in projections:
            # transform projection to (BH x T x E)
            output_projections.append(
                projection.view(
                    batch_size,
                    -1,
                    self.num_heads,
                    self.projection_dim
                ).transpose(2, 1).contiguous().view(
                    batch_size * self.num_heads,
                    -1,
                    self.projection_dim
                )
            )

        return output_projections

    def attention(self, queries, keys, values):
        ''' Scaled dot product attention with optional masks '''
        logits = self.scale * torch.bmm(queries, keys.transpose(2, 1))

        # attended = torch.bmm(F.softmax(logits, dim=-1), values)
        attn_weights = self.sparsemax(logits)
        return attn_weights

    def forward(self, queries, keys, values):
        ''' Forward pass of the attention '''
        # pylint:disable=unbalanced-tuple-unpacking
        if same_tensor(values, keys, queries):
            values, keys, queries = self.project(values, chunks=3)
        elif same_tensor(values, keys):
            values, keys = self.project(values, chunks=2)
            queries, = self.project(queries, 2)
        else:
            values, = self.project(values, 0)
            keys, = self.project(keys, 1)
            queries, = self.project(queries, 2)

        attn_weights = self.attention(queries, keys, values)
        return attn_weights


class GroupGenerateLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, scale=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale = scale
       
        self.multihead_attns = nn.ModuleList([SparseMultiHeadedAttention(hidden_dim, scale=scale) \
                                                for _ in range(num_heads)])

    def forward(self, query, key_value, epoch=0):
        """
            Use multiheaded attention to get mask
            Num_interpretable_heads = num_heads * seq_len
            Input: x (bsz, seq_len, hidden_dim)
                   if actual_x is not None, then use actual_x instead of x to compute attn_output
            Output: attn_outputs (bsz, num_heads * seq_len, seq_len, hidden_dim)
                    mask (bsz, num_heads, seq_len, seq_len)
        """

        if epoch == -1:
            epoch = self.num_heads
        
        head_i = epoch % self.num_heads
        if self.training:
            attn_weights = self.multihead_attns[head_i](query, key_value, key_value)
        else:
            attn_weights = []
            if epoch < self.num_heads:
                num_heads_use = head_i + 1
            else:
                num_heads_use = self.num_heads
            for head_j in range(num_heads_use):
                attn_weights_j = self.multihead_attns[head_j](query, key_value, key_value)
                attn_weights.append(attn_weights_j)
            attn_weights = torch.stack(attn_weights, dim=1)
        # import pdb; pdb.set_trace()
        # attn_weights = 
        return attn_weights


class GroupSelectLayer(nn.Module):
    def __init__(self, hidden_dim, scale=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.multihead_attn = SparseMultiHeadedAttention(hidden_dim, scale=scale)
        self.scale = scale

    def forward(self, query, key, value):
        """
            Use multiheaded attention to get mask
            Num_heads = num_heads * seq_len
            Input: x (bsz, seq_len, hidden_dim)
                   if actual_x is not None, then use actual_x instead of x to compute attn_output
            Output: attn_outputs (bsz, num_heads * seq_len, seq_len, hidden_dim)
                    mask (bsz, num_heads, seq_len, seq_len)
        """
        # x shape: (batch_size, sequence_length, hidden_dim)
        # x shape: (..., hidden_dim)
        epsilon = 1e-30
        bsz, seq_len, hidden_dim = query.shape

        # Obtain attention weights
        attn_weights = self.multihead_attn(query, key, key)
        mask = attn_weights.transpose(-1, -2)

        # Apply attention weights on what to be attended
        new_shape = list(mask.shape) + [1] * (len(value.shape) - 3)
        attn_outputs = (value * mask.view(*new_shape)).sum(1)

        # attn_outputs of shape (bsz, num_masks, num_classes)
        return attn_outputs, mask


class SOPConfig:
    def __init__(self,
                 json_file=None,
                 hidden_size: int = None,
                 input_hidden_size: int = None,
                 num_labels: int = None,
                 projected_input_scale: int = None,
                 num_heads: int = None,
                 num_masks_sample: int = None,
                 num_masks_max: int = None,
                 image_size=None,
                 num_channels: int = None,
                 attn_patch_size: int = None,
                 finetune_layers=None,
                 group_gen_scale: int = None,
                 group_gen_temp_alpha: int = None,
                 group_gen_temp_beta: int = None,
                 group_sel_scale: int = None,
                 group_gen_blur_ks1: int = None,
                 group_gen_blur_sigma1: int = None,
                 group_gen_blur_ks2: int = None,
                 group_gen_blur_sigma2: int = None,
                #  group_gen_scale_sigmoid: int = None,
                #  group_gen_scale_sigmoid_log: int = None,
                #  group_gen_scale_sigmoid_mode: str = None,
                 freeze_projection: str = None
                ):
        # all the config from the json file will be in self.__dict__
        super().__init__()

        # set default values
        self.hidden_size = 512
        self.input_hidden_size = None
        self.num_labels = 2
        self.projected_input_scale = 1
        self.num_heads = 1
        self.num_masks_sample = 20
        self.num_masks_max = 200
        self.image_size=(224, 224)
        self.num_channels = 3
        self.attn_patch_size = 16
        self.finetune_layers=[]
        self.group_gen_scale = 1
        self.group_gen_temp_alpha = 1
        self.group_gen_temp_beta = 1
        self.group_sel_scale = 1
        self.group_gen_blur_ks1 = -1
        self.group_gen_blur_sigma1 = -1
        self.group_gen_blur_ks2 = -1
        self.group_gen_blur_sigma2 = -1
        # self.group_gen_scale_sigmoid = -1
        # self.group_gen_scale_sigmoid_log = -1
        # self.group_gen_scale_sigmoid_mode = 'median' # or '0.5'
        self.freeze_projection = False

        # first load the config from json file if specified
        if json_file is not None:
            self.update_from_json(json_file)
        
        # overwrite the config from json file if specified
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if input_hidden_size is not None:
            self.input_hidden_size = input_hidden_size
        if num_labels is not None: 
            self.num_labels = num_labels
        if projected_input_scale is not None:
            self.projected_input_scale = projected_input_scale
        if num_heads is not None:
            self.num_heads = num_heads
        if num_masks_sample is not None:
            self.num_masks_sample = num_masks_sample
        if num_masks_max is not None:
            self.num_masks_max = num_masks_max
        if image_size is not None:
            self.image_size = image_size
        if num_channels is not None:
            self.num_channels = num_channels
        if attn_patch_size is not None:
            self.attn_patch_size = attn_patch_size
        if finetune_layers is not None:
            self.finetune_layers = finetune_layers
        if group_gen_scale is not None:
            self.group_gen_scale = group_gen_scale
        if group_gen_temp_alpha is not None:
            self.group_gen_temp_alpha = group_gen_temp_alpha
            assert group_gen_temp_alpha > 0
        if group_gen_temp_beta is not None:
            self.group_gen_temp_beta = group_gen_temp_beta
            assert group_gen_temp_beta > 0
        if group_sel_scale is not None:
            self.group_sel_scale = group_sel_scale
        if group_gen_blur_ks1 is not None:
            self.group_gen_blur_ks1 = group_gen_blur_ks1
        if group_gen_blur_sigma1 is not None:
            self.group_gen_blur_sigma1 = group_gen_blur_sigma1
        if group_gen_blur_ks2 is not None:
            self.group_gen_blur_ks2 = group_gen_blur_ks2
        if group_gen_blur_sigma2 is not None:
            self.group_gen_blur_sigma2 = group_gen_blur_sigma2
        # if group_gen_scale_sigmoid is not None:
        #     self.group_gen_scale_sigmoid = group_gen_scale_sigmoid
        # if group_gen_scale_sigmoid_log is not None:
        #     self.group_gen_scale_sigmoid_log = group_gen_scale_sigmoid_log
        # if group_gen_scale_sigmoid_mode is not None:
        #     self.group_gen_scale_sigmoid_mode = group_gen_scale_sigmoid_mode
        if freeze_projection is not None:
            self.freeze_projection = freeze_projection

        
    def update_from_json(self, json_file):
        with open(json_file, 'r') as f:
            json_dict = json.load(f)
        self.__dict__.update(json_dict)

    def save_to_json(self, json_file):
        attrs_save = [
            'hidden_size',
            'input_hidden_size',
            'num_labels',
            'projected_input_scale',
            'num_heads',
            'num_masks_sample',
            'num_masks_max',
            'image_size',
            'num_channels',
            'attn_patch_size',
            'finetune_layers',
            'group_gen_scale',
            'group_gen_temp_alpha',
            'group_gen_temp_beta',
            'group_sel_scale',
            'group_gen_blur_ks1',
            'group_gen_blur_sigma1',
            'group_gen_blur_ks2',
            'group_gen_blur_sigma2',
            # 'group_gen_scale_sigmoid',
            # 'group_gen_scale_sigmoid_log',
            # 'group_gen_scale_sigmoid_mode',
            'freeze_projection'
        ]
        to_save = {k: v for k, v in self.__dict__.items() if k in attrs_save}
        with open(json_file, 'w') as f:
            json.dump(to_save, f, indent=4)


class SOP(nn.Module):
    config_class = SOPConfig

    def __init__(self, 
                 config,
                 backbone_model,
                 class_weights=None,
                 ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # match black_box_model hidden_size
        self.input_hidden_size = config.input_hidden_size if (hasattr(config, 'input_hidden_size') is not None and config.input_hidden_size is not None) else self.hidden_size
        self.num_classes = config.num_labels if hasattr(config, 'num_labels') is not None else 1  # 1 is for regression
        self.projected_input_scale = config.projected_input_scale if hasattr(config, 'projected_input_scale') else 1
        self.num_heads = config.num_heads
        self.num_masks_sample = config.num_masks_sample
        self.num_masks_max = config.num_masks_max
        self.finetune_layers = config.finetune_layers
        self.group_gen_scale = config.group_gen_scale if hasattr(config, 'group_gen_scale') else 1
        self.group_gen_temp_alpha = config.group_gen_temp_alpha \
            if hasattr(config, 'group_gen_temp_alpha') else 1
        self.group_gen_temp_beta = config.group_gen_temp_beta if hasattr(config, 'group_gen_temp_beta') else 1
        # self.group_gen_scale_sigmoid = config.group_gen_scale_sigmoid if hasattr(config, 'group_gen_scale_sigmoid') else -1
        # self.group_gen_scale_sigmoid_log = config.group_gen_scale_sigmoid_log if hasattr(config, 'group_gen_scale_sigmoid_log') else -1
        self.group_sel_scale = config.group_sel_scale if hasattr(config, 'group_sel_scale') else 1
        # self.group_gen_scale_sigmoid_mode = config.group_gen_scale_sigmoid_mode if hasattr(config, 'group_gen_scale_sigmoid_mode') else 'median'
        self.group_gen_blur_ks1 = config.group_gen_blur_ks1 if hasattr(config, 'group_gen_blur_ks1') else -1
        self.group_gen_blur_sigma1 = config.group_gen_blur_sigma1 if hasattr(config, 'group_gen_blur_sigma1') else -1
        self.group_gen_blur_ks2 = config.group_gen_blur_ks2 if hasattr(config, 'group_gen_blur_ks2') else -1
        self.group_gen_blur_sigma2 = config.group_gen_blur_sigma2 if hasattr(config, 'group_gen_blur_sigma2') else -1

        # blackbox model and finetune layers
        self.blackbox_model = backbone_model
        if class_weights is None:
            try:
                class_weights = get_chained_attr(backbone_model, config.finetune_layers[0]).weight
            except:
                raise ValueError('class_weights is None and cannot be inferred from backbone_model')
        # try:
        #     self.class_weights = copy.deepcopy(class_weights)  # maybe can do this outside
        #     print('deep copy class weights')
        # except:
        self.class_weights = nn.Parameter(class_weights.clone())
            # print('shallow copy class weights')
        
        
        self.input_attn = GroupGenerateLayer(hidden_dim=self.input_hidden_size,
                                             num_heads=self.num_heads,
                                             scale=self.group_gen_scale)
        # output
        self.output_attn = GroupSelectLayer(hidden_dim=self.hidden_size,
                                            scale=self.group_sel_scale)

    def init_grads(self):
        # Initialize the weights of the model
        # blackbox_model = self.blackbox_model.clone()
        # self.init_weights()
        # self.blackbox_model = blackbox_model

        if hasattr(self.config, 'freeze_projection') and (self.config.freeze_projection):
            for name, param in self.projection.named_parameters():
                param.requires_grad = False
            print('projection layer is frozen')
        else:
            print('projection layer is not frozen')

        for name, param in self.blackbox_model.named_parameters():
            param.requires_grad = False

        # for finetune_layer in self.finetune_layers:
        #     # todo: check if this works with index
        #     for name, param in get_chained_attr(self.blackbox_model, finetune_layer).named_parameters():
        #         param.requires_grad = True

    def forward(self):
        raise NotImplementedError
    
    def save(self, save_dir):
        self.config.save_to_json(os.path.join(save_dir, 'config.json'))
        torch.save(self.state_dict(), os.path.join(save_dir, 'model.pt'))
        print('Saved model to {}'.format(save_dir))

    def load(self, save_dir):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config.update_from_json(os.path.join(save_dir, 'config.json'))
        self.load_state_dict(torch.load(os.path.join(save_dir, 'model.pt'), map_location=device))
        print('Loaded model from {}'.format(save_dir))

    def load_checkpoint(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path)['model'])
        print('Loaded model from checkpoint {}'.format(checkpoint_path))


class SOPImage(SOP):
    def __init__(self, 
                 config,
                 blackbox_model,
                 class_weights=None,
                 projection_layer=None,
                 ):
        super().__init__(config,
                            blackbox_model,
                            class_weights
                            )
        
        self.image_size = config.image_size if isinstance(config.image_size, 
                                                    collections.abc.Iterable) \
                                            else (config.image_size, config.image_size)
        self.num_channels = config.num_channels
        # attention args
        self.attn_patch_size = config.attn_patch_size

        if projection_layer is not None:
            self.projection = copy.deepcopy(projection_layer)
        else:
            self.init_projection()

        # self.projection_up = nn.ConvTranspose2d(1, 
        #                                             1, 
        #                                             kernel_size=self.attn_patch_size, 
        #                                             stride=self.attn_patch_size)  # make each patch a vec
        # self.projection_up.weight = nn.Parameter(torch.ones_like(self.projection_up.weight))
        # self.projection_up.bias = nn.Parameter(torch.zeros_like(self.projection_up.bias))
        # self.projection_up.weight.requires_grad = False
        # self.projection_up.bias.requires_grad = False

        # Initialize the weights of the model
        # self.init_weights()
        self.init_grads()

    def init_projection(self):
        self.projection = nn.Conv2d(self.config.num_channels, 
                                    self.input_hidden_size, 
                                    kernel_size=self.attn_patch_size, 
                                    stride=self.attn_patch_size)  # make each patch a vec
        
    def forward(self, 
                inputs, 
                segs=None, 
                input_mask_weights=None,
                epoch=-1, 
                mask_batch_size=16,
                label=None,
                return_tuple=False,
                binary_threshold=-1):
        if epoch == -1:
            epoch = self.num_heads
        bsz, num_channel, img_dim1, img_dim2 = inputs.shape
        
        # Mask (Group) generation
        if input_mask_weights is None:
            grouped_inputs, input_mask_weights = self.group_generate(inputs, epoch, mask_batch_size, segs, binary_threshold)
        else:
            grouped_inputs = inputs.unsqueeze(1) * input_mask_weights.unsqueeze(2) # directly apply mask

        # Backbone model
        logits, pooler_outputs = self.run_backbone(grouped_inputs, mask_batch_size)
        # return logits

        # Mask (Group) selection & aggregation
        # return self.group_select(logits, pooler_outputs, img_dim1, img_dim2)
        weighted_logits, output_mask_weights, logits, pooler_outputs = self.group_select(logits, pooler_outputs, img_dim1, img_dim2)

        if return_tuple:
            return self.get_results_tuple(weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label)
        else:
            return weighted_logits

    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label):
        raise NotImplementedError

    def run_backbone(self, masked_inputs, mask_batch_size):
        bsz, num_masks, num_channel, img_dim1, img_dim2 = masked_inputs.shape
        masked_inputs = masked_inputs.view(-1, num_channel, img_dim1, img_dim2)
        logits = []
        pooler_outputs = []
        for i in range(0, masked_inputs.shape[0], mask_batch_size):
            output_i = self.blackbox_model(
                masked_inputs[i:i+mask_batch_size]
            )
            pooler_i = output_i.pooler_output
            logits_i = output_i.logits
            logits.append(logits_i)
            pooler_outputs.append(pooler_i)

        logits = torch.cat(logits).view(bsz, num_masks, self.num_classes, -1)
        pooler_outputs = torch.cat(pooler_outputs).view(bsz, num_masks, self.hidden_size, -1)
        return logits, pooler_outputs
    
    def group_generate(self, inputs, epoch, mask_batch_size, segs=None, binary_threshold=-1):
        bsz, num_channel, img_dim1, img_dim2 = inputs.shape
        if segs is None:   # should be renamed "segments"
            projected_inputs = self.projection(inputs)
            num_patches = projected_inputs.shape[-2:]
            projected_inputs = projected_inputs.flatten(2).transpose(1, 2)  # bsz, img_dim1 * img_dim2, num_channel
            projected_inputs = projected_inputs * self.projected_input_scale

            if self.num_masks_max != -1:
                input_dropout_idxs = torch.randperm(projected_inputs.shape[1])[:self.num_masks_max]
                projected_query = projected_inputs[:, input_dropout_idxs]
            else:
                projected_query = projected_inputs
            input_mask_weights_cand = self.input_attn(projected_query, projected_inputs, epoch=epoch)
            
            # num_patches = ((self.image_size[0] - self.attn_patch_size) // self.attn_patch_size + 1, 
            #             (self.image_size[1] - self.attn_patch_size) // self.attn_patch_size + 1)
            input_mask_weights_cand = input_mask_weights_cand.reshape(-1, 1, num_patches[0], num_patches[1])
            input_mask_weights_cand = torch.nn.functional.interpolate(input_mask_weights_cand, size=(img_dim1, img_dim2), mode='nearest')
            # input_mask_weights_cand = self.projection_up(input_mask_weights_cand, 
            #                                              output_size=torch.Size([input_mask_weights_cand.shape[0], 1, 
            #                                                                      img_dim1, img_dim2]))
            input_mask_weights_cand = input_mask_weights_cand.view(bsz, -1, img_dim1, img_dim2)
            input_mask_weights_cand = torch.clip(input_mask_weights_cand, max=1.0)
        else:
            # With/without masks are a bit different. Should we make them the same? Need to experiment.
            bsz, num_segs, img_dim1, img_dim2 = segs.shape
            seged_output_0 = inputs.unsqueeze(1) * segs.unsqueeze(2) # (bsz, num_masks, num_channel, img_dim1, img_dim2)
            # import pdb; pdb.set_trace()
            _, interm_outputs = self.run_backbone(seged_output_0, mask_batch_size)
            
            interm_outputs = interm_outputs.view(bsz, -1, self.hidden_size)
            interm_outputs = interm_outputs * self.projected_input_scale
            segment_mask_weights = self.input_attn(interm_outputs, interm_outputs, epoch=epoch)
            segment_mask_weights = segment_mask_weights.reshape(bsz, -1, num_segs)
            
            new_masks =  segs.unsqueeze(1) * segment_mask_weights.unsqueeze(-1).unsqueeze(-1)
            # (bsz, num_new_masks, num_masks, img_dim1, img_dim2)
            input_mask_weights_cand = new_masks.sum(2)  # if one mask has it, then have it
            # todo: Can we simplify the above to be dot product?
            
        scale_factor = 1.0 / input_mask_weights_cand.reshape(bsz, -1, 
                                                        img_dim1 * img_dim2).max(dim=-1).values
        input_mask_weights_cand = input_mask_weights_cand * scale_factor.view(bsz, -1,1,1)

        if self.group_gen_blur_ks1 != -1 and self.group_gen_blur_sigma1 != -1:
            input_mask_weights_cand = gaussian_blur(input_mask_weights_cand, 
                                                    self.group_gen_blur_ks1, 
                                                    self.group_gen_blur_sigma1)

        # temperature scaling to make the mask weights closer to 0 or 1
        input_mask_weights_cand = torch.sigmoid(torch.log(input_mask_weights_cand / \
            self.group_gen_temp_beta + EPS) / self.group_gen_temp_alpha)

        if self.group_gen_blur_ks2 != -1 and self.group_gen_blur_sigma2 != -1:
            input_mask_weights_cand = gaussian_blur(input_mask_weights_cand, 
                                                    self.group_gen_blur_ks2, 
                                                    self.group_gen_blur_sigma2)

        if binary_threshold != -1 and not self.training: # if binary threshold is set, then use binary mask above the threshold only for testing
            input_mask_weights_cand = (input_mask_weights_cand > binary_threshold).float()

        # print('self.group_gen_temperature', self.group_gen_temperature)
        
        # if self.group_gen_scale_sigmoid != -1:
        #     # import pdb; pdb.set_trace()
        #     if self.group_gen_scale_sigmoid_mode == 'median':
        #         input_mask_weights_cand = torch.sigmoid((input_mask_weights_cand - input_mask_weights_cand.median())/(input_mask_weights_cand.max() - input_mask_weights_cand.min()) * self.group_gen_scale_sigmoid)
        #     elif self.group_gen_scale_sigmoid_mode == 'min':
        #         input_mask_weights_cand = torch.sigmoid(((input_mask_weights_cand - input_mask_weights_cand.min())/ \
        #                 (input_mask_weights_cand.max() - input_mask_weights_cand.min()) - 0.5) * self.group_gen_scale_sigmoid)
        #     elif self.group_gen_scale_sigmoid_mode == '0.5':
        #         input_mask_weights_cand = torch.sigmoid((input_mask_weights_cand - 0.5) * self.group_gen_scale_sigmoid)
        #     elif self.group_gen_scale_sigmoid_mode == 'log':
        #         input_mask_weights_cand = torch.sigmoid(torch.log(input_mask_weights_cand * self.group_gen_scale_sigmoid_log + 1e-8) * self.group_gen_scale_sigmoid)
            # else: # none
            #     raise ValueError('group_gen_scale_sigmoid_mode should be either median or min or 0.5 or log')
        
        # we are using iterative training
        # we will train some masks every epoch
        # the masks to train are selected by mod of epoch number
        # Dropout for training
        if self.training:
            dropout_idxs = torch.randperm(input_mask_weights_cand.shape[1])[:self.num_masks_sample]
            dropout_mask = torch.zeros(bsz, input_mask_weights_cand.shape[1]).to(inputs.device)
            dropout_mask[:,dropout_idxs] = 1
        else:
            dropout_mask = torch.ones(bsz, input_mask_weights_cand.shape[1]).to(inputs.device)
        
        input_mask_weights = input_mask_weights_cand[dropout_mask.bool()].clone()
        input_mask_weights = input_mask_weights.view(bsz, -1, img_dim1, img_dim2)

        masked_inputs = inputs.unsqueeze(1) * input_mask_weights.unsqueeze(2)
        return masked_inputs, input_mask_weights
    
    def group_select(self, logits, pooler_outputs, img_dim1, img_dim2):
        raise NotImplementedError

    def get_masks_used(self, outputs, i=0):
        pred = outputs.logits[i].argmax(-1).item()
        pred_mask_idxs_sort = outputs.mask_weights[i,:,pred].argsort(descending=True)
        mask_weights_sort = (outputs.mask_weights * outputs.logits_all)[i,pred_mask_idxs_sort,pred]
        masks_sort = outputs.masks[0,pred_mask_idxs_sort]
        masks_sort_used = (masks_sort[mask_weights_sort != 0] > masks_sort[mask_weights_sort != 0].mean()).int()
        mask_weights_sort_used = mask_weights_sort[mask_weights_sort != 0]
        return {
            'masks_sort_used': masks_sort_used, 
            'mask_weights_sort_used': mask_weights_sort_used
        }
        

class SOPImageCls(SOPImage):
    def group_select(self, logits, pooler_outputs, img_dim1, img_dim2):
        bsz, num_masks = logits.shape[:2]

        logits = logits.view(bsz, num_masks, self.num_classes)
        pooler_outputs = pooler_outputs.view(bsz, num_masks, self.hidden_size)

        query = self.class_weights.unsqueeze(0).expand(bsz, 
                                                    self.num_classes, 
                                                    self.hidden_size) #.to(logits.device)
        
        key = pooler_outputs
        weighted_logits, output_mask_weights = self.output_attn(query, key, logits)

        return weighted_logits, output_mask_weights, logits, pooler_outputs
    
    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label):
        # todo: debug for segmentation
        masks_aggr = None
        masks_aggr_pred_cls = None
        masks_max_pred_cls = None
        flat_masks = None

        if label is not None:
            predicted = label  # allow labels to be different
        else:
            _, predicted = torch.max(weighted_logits.data, -1)
        
        grouped_attributions = output_mask_weights * logits # instead of just output_mask_weights
        # import pdb; pdb.set_trace()
        masks_mult_pred = input_mask_weights * grouped_attributions[range(len(predicted)),:,predicted,None,None]
        masks_aggr_pred_cls = masks_mult_pred.sum(1)[:,None,:,:]
        max_mask_indices = grouped_attributions.max(2).values.max(1).indices
        # import pdb; pdb.set_trace()
        masks_max_pred_cls = masks_mult_pred[range(bsz),max_mask_indices]

        flat_masks = compress_masks_image(input_mask_weights, grouped_attributions[:,:,predicted])
        
        return AttributionOutputSOP(weighted_logits,
                                    logits,
                                    pooler_outputs,
                                    input_mask_weights,
                                    output_mask_weights,
                                    masks_aggr_pred_cls,
                                    masks_max_pred_cls,
                                    masks_aggr,
                                    flat_masks,
                                    grouped_attributions)



def default_projection_func(blackbox_model, inputs):
    # This is an example projection function. Customize as needed.
    projected_inputs = blackbox_model.model(inputs, output_hidden_states=True).hidden_states[-2][:,1:].transpose(1,2)
    num_patch = 14  # Example patch size. Customize as needed.
    projected_inputs = projected_inputs.reshape(-1, 
                                                blackbox_model.model.config.hidden_size, num_patch, num_patch)
    return projected_inputs


class SOPImageCls2(SOPImageCls):
    # def __init__(self, 
    #              config,
    #              blackbox_model,
    #              class_weights=None,
    #              projection_layer=None,
    #              ):
    #     super().__init__(config,
    #                         blackbox_model,
    #                         class_weights,
    #                         projection_layer
    #                         )
    def __init__(self, 
                 config,
                 blackbox_model,
                 class_weights=None,
                 projection_layer=None,
                 projection_func=default_projection_func
                 ):
        super().__init__(config,
                            blackbox_model,
                            class_weights
                            )
        self.group_gen_blur_ks1 = config.group_gen_blur_ks1
        self.group_gen_blur_sigma1 = config.group_gen_blur_sigma1
        
        self.input_attn = GroupGenerateLayerBlur(hidden_dim=self.input_hidden_size,
                                             num_heads=self.num_heads,
                                             scale=self.group_gen_scale,
                                             kernel_size=config.group_gen_blur_ks1,
                                             sigma=config.group_gen_blur_sigma1
                                                )
        
        self.image_size = config.image_size if isinstance(config.image_size, 
                                                    collections.abc.Iterable) \
                                            else (config.image_size, config.image_size)
        self.num_channels = config.num_channels
        # attention args
        self.attn_patch_size = config.attn_patch_size
        
        self.init_grads()
        
        for name, param in self.projection.named_parameters():
            param.requires_grad = False
            
        self.projection_func = projection_func
        
    def group_generate(self, inputs, epoch, mask_batch_size, segs=None, binary_threshold=-1):
        bsz, num_channel, img_dim1, img_dim2 = inputs.shape
        if segs is None:   # should be renamed "segments"
            # projected_inputs = self.projection(inputs)
            # new
            num_patch = 14
            # TODO: how to write this in a general way? 
            # I would like to pass it in as a function, but I need it to rely on blackbox_model,
            # which does not always have the processing steps like this.
            # projected_inputs = self.blackbox_model.model(inputs, output_hidden_states=True).hidden_states[-2][:,1:].transpose(1,2).reshape(-1, 
            #                 self.config.hidden_size, num_patch, num_patch)
            projected_inputs = self.projection_func(self.blackbox_model, inputs)
            # new over
            num_patches = projected_inputs.shape[-2:]
            projected_inputs = projected_inputs.flatten(2).transpose(1, 2)  # bsz, img_dim1 * img_dim2, num_channel
            projected_inputs = projected_inputs * self.projected_input_scale

            if self.num_masks_max != -1:
                input_dropout_idxs = torch.randperm(projected_inputs.shape[1])[:self.num_masks_max]
                projected_query = projected_inputs[:, input_dropout_idxs]
            else:
                projected_query = projected_inputs
            input_mask_weights_cand = self.input_attn(projected_query, projected_inputs, epoch=epoch)
            
            input_mask_weights_cand = input_mask_weights_cand.reshape(-1, 1, num_patches[0], num_patches[1])
            input_mask_weights_cand = torch.nn.functional.interpolate(input_mask_weights_cand, size=(img_dim1, img_dim2), mode='nearest')
            
            input_mask_weights_cand = input_mask_weights_cand.view(bsz, -1, img_dim1, img_dim2)
            # input_mask_weights_cand = torch.clip(input_mask_weights_cand, max=1.0)
        else:
            # With/without masks are a bit different. Should we make them the same? Need to experiment.
            bsz, num_segs, img_dim1, img_dim2 = segs.shape
            seged_output_0 = inputs.unsqueeze(1) * segs.unsqueeze(2) # (bsz, num_masks, num_channel, img_dim1, img_dim2)
            # import pdb; pdb.set_trace()
            _, interm_outputs = self.run_backbone(seged_output_0, mask_batch_size)
            
            interm_outputs = interm_outputs.view(bsz, -1, self.hidden_size)
            interm_outputs = interm_outputs * self.projected_input_scale
            segment_mask_weights = self.input_attn(interm_outputs, interm_outputs, epoch=epoch)
            segment_mask_weights = segment_mask_weights.reshape(bsz, -1, num_segs)
            
            new_masks =  segs.unsqueeze(1) * segment_mask_weights.unsqueeze(-1).unsqueeze(-1)
            # (bsz, num_new_masks, num_masks, img_dim1, img_dim2)
            input_mask_weights_cand = new_masks.sum(2)  # if one mask has it, then have it
            # todo: Can we simplify the above to be dot product?
            
            
        scale_factor = 1.0 / input_mask_weights_cand.reshape(bsz, -1, 
                                                        img_dim1 * img_dim2).max(dim=-1).values
        input_mask_weights_cand = input_mask_weights_cand * scale_factor.view(bsz, -1,1,1)
        
        if self.training:
            dropout_idxs = torch.randperm(input_mask_weights_cand.shape[1])[:self.num_masks_sample]
            dropout_mask = torch.zeros(bsz, input_mask_weights_cand.shape[1]).to(inputs.device)
            dropout_mask[:,dropout_idxs] = 1
        else:
            dropout_mask = torch.ones(bsz, input_mask_weights_cand.shape[1]).to(inputs.device)
        
        input_mask_weights = input_mask_weights_cand[dropout_mask.bool()].clone()
        input_mask_weights = input_mask_weights.view(bsz, -1, img_dim1, img_dim2)

        masked_inputs = inputs.unsqueeze(1) * input_mask_weights.unsqueeze(2)
        # import pdb; pdb.set_trace()
        # print('input_mask_weights', input_mask_weights.min(), input_mask_weights.max())
        return masked_inputs, input_mask_weights
    

class SOPImageSeg(SOPImage):
    def group_select(self, logits, pooler_outputs, img_dim1, img_dim2):
        bsz, num_masks = logits.shape[:2]
        logits = logits.view(bsz, num_masks, self.num_classes, 
                                            img_dim1, img_dim2)
        pooler_outputs = pooler_outputs.view(bsz, num_masks, 
                                                        self.hidden_size, 
                                                        img_dim1, 
                                                        img_dim2)
        # return pooler_outputs
        query = self.class_weights.unsqueeze(0) \
            .view(1, self.num_classes, self.hidden_size, -1).mean(-1) \
            .expand(bsz, self.num_classes, self.hidden_size).to(logits.device)
        pooler_outputs.requires_grad = True
        key = pooler_outputs.view(bsz, num_masks, self.hidden_size, -1).mean(-1)
        weighted_logits, output_mask_weights = self.output_attn(query, key, logits)

        return weighted_logits, output_mask_weights, logits, pooler_outputs
    
    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label):
        # todo: debug for segmentation
        masks_aggr = None
        masks_aggr_pred_cls = None
        masks_max_pred_cls = None
        flat_masks = None
        grouped_attributions = None

        # import pdb
        # pdb.set_trace()
        _, predicted = torch.max(weighted_logits.data, -1)
        masks_mult = input_mask_weights.unsqueeze(2) * output_mask_weights.unsqueeze(-1).unsqueeze(-1) # bsz, n_masks, n_cls, img_dim, img_dim
        masks_aggr = masks_mult.sum(1) # bsz, n_cls, img_dim, img_dim OR bsz, n_cls, seq_len
        # masks_aggr_pred_cls = masks_aggr
        # masks_aggr_pred_cls = masks_aggr[range(bsz), predicted].unsqueeze(1)
        # max_mask_indices = output_mask_weights.argmax(1)
        # masks_max_pred_cls = masks_mult[max_mask_indices[:,0]]
        # masks_max_pred_cls = max_mask_indices
        # TODO: this has some problems ^
        # import pdb
        # pdb.set_trace()
        # grouped_attributions = output_mask_weights * logits
        
        # flat_masks = compress_masks_image(input_mask_weights, output_mask_weights[:,:,predicted])

        return AttributionOutputSOP(weighted_logits,
                                    logits,
                                    pooler_outputs,
                                    input_mask_weights,
                                    output_mask_weights,
                                    masks_aggr_pred_cls,
                                    masks_max_pred_cls,
                                    masks_aggr,
                                    flat_masks,
                                    grouped_attributions)


def gaussian_kernel(size, sigma, device):
    """Generates a 2D Gaussian kernel."""
    coords = torch.tensor([(x - size // 2) for x in range(size)]).to(device)
    grid = coords.unsqueeze(0).repeat(size, 1)
    kernel = torch.exp(-(grid ** 2 + grid.t() ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel

def gaussian_blur(image, kernel_size=5, sigma=1):
    """Applies Gaussian blur to an image."""
    channels = image.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, image.device)
    # Reshape to 2d depthwise convolutional weight
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = kernel_size // 2
    blurred_image = F.conv2d(image, kernel, padding=padding, groups=channels)
    return blurred_image


class SparseMultiHeadedAttentionBlur(nn.Module):
    ''' Implement a multi-headed attention module '''
    def __init__(self, embed_dim, num_heads=1, scale=1, kernel_size=1, sigma=1):
        ''' Initialize the attention module '''
        super().__init__()

        # ensure valid inputs
        assert embed_dim % num_heads == 0, \
            f'num_heads={num_heads} should evenly divide embed_dim={embed_dim}'

        # store off the scale and input params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.scale = scale
        self.kernel_size = kernel_size
        self.sigma = sigma
        # self.scale = self.projection_dim ** -0.5

        # Combine projections for multiple heads into a single linear layer for efficiency
        self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sparsemax = Sparsemax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialization '''
        # Initialize using Xavier
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.input_weights, gain)
        nn.init.xavier_uniform_(self.output_projection.weight, gain)

    def project(self, inputs, index=0, chunks=1):
        ''' Produce a linear projection using the weights '''
        batch_size = inputs.shape[0]
        start = index * self.embed_dim
        end = start + chunks * self.embed_dim
        # import pdb; pdb.set_trace()
        projections = F.linear(inputs, self.input_weights[start:end]).chunk(chunks, dim=-1)

        output_projections = []
        for projection in projections:
            # transform projection to (BH x T x E)
            output_projections.append(
                projection.view(
                    batch_size,
                    -1,
                    self.num_heads,
                    self.projection_dim
                ).transpose(2, 1).contiguous().view(
                    batch_size * self.num_heads,
                    -1,
                    self.projection_dim
                )
            )

        return output_projections

    def attention(self, queries, keys, values):
        ''' Scaled dot product attention with optional masks '''
        logits = self.scale * torch.bmm(queries, keys.transpose(2, 1))

        # attended = torch.bmm(F.softmax(logits, dim=-1), values)
        # print('logits', logits.shape)
        # import pdb; pdb.set_trace()
        num_patches = int(math.sqrt(logits.shape[-1]))
        bsz, num_groups, _ = logits.shape
        logits_reshape = logits.view(bsz, num_groups, num_patches, num_patches)
        # print('logits_reshape', logits_reshape.shape)
        logits_reshape_blurred = gaussian_blur(logits_reshape, self.kernel_size, self.sigma)
        # print('logits_reshape_blurred', logits_reshape_blurred.shape)
        attn_weights = self.sparsemax(logits_reshape_blurred.flatten(-2))
        # attn_weights_raw = self.sparsemax(logits)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(attn_weights[0][0].view(num_patches, num_patches).cpu())
        # plt.show()
        # plt.figure()
        # plt.imshow(attn_weights_raw[0][0].view(num_patches, num_patches).cpu())
        # plt.show()
        # import pdb; pdb.set_trace()
        return attn_weights

    def forward(self, queries, keys, values):
        ''' Forward pass of the attention '''
        # pylint:disable=unbalanced-tuple-unpacking
        if same_tensor(values, keys, queries):
            values, keys, queries = self.project(values, chunks=3)
        elif same_tensor(values, keys):
            values, keys = self.project(values, chunks=2)
            queries, = self.project(queries, 2)
        else:
            values, = self.project(values, 0)
            keys, = self.project(keys, 1)
            queries, = self.project(queries, 2)

        attn_weights = self.attention(queries, keys, values)
        return attn_weights


class GroupGenerateLayerBlur(nn.Module):
    def __init__(self, hidden_dim, num_heads, scale=1, kernel_size=1, sigma=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale = scale
        self.kernel_size = kernel_size
        self.sigma = sigma
       
        self.multihead_attns = nn.ModuleList([SparseMultiHeadedAttentionBlur(hidden_dim, 
                                                                             scale=scale,
                                                                             kernel_size=kernel_size,
                                                                             sigma=sigma) \
                                                for _ in range(num_heads)])

    def forward(self, query, key_value, epoch=0):
        """
            Use multiheaded attention to get mask
            Num_interpretable_heads = num_heads * seq_len
            Input: x (bsz, seq_len, hidden_dim)
                   if actual_x is not None, then use actual_x instead of x to compute attn_output
            Output: attn_outputs (bsz, num_heads * seq_len, seq_len, hidden_dim)
                    mask (bsz, num_heads, seq_len, seq_len)
        """

        if epoch == -1:
            epoch = self.num_heads
        
        head_i = epoch % self.num_heads
        if self.training:
            attn_weights = self.multihead_attns[head_i](query, key_value, key_value)
        else:
            attn_weights = []
            if epoch < self.num_heads:
                num_heads_use = head_i + 1
            else:
                num_heads_use = self.num_heads
            for head_j in range(num_heads_use):
                attn_weights_j = self.multihead_attns[head_j](query, key_value, key_value)
                attn_weights.append(attn_weights_j)
            attn_weights = torch.stack(attn_weights, dim=1)
        # import pdb; pdb.set_trace()
        # attn_weights = 
        return attn_weights


# neurips

class SparseMultiHeadedAttentionBlur(nn.Module):
    ''' Implement a multi-headed attention module '''
    def __init__(self, embed_dim, num_heads=1, scale=1, kernel_size=1, sigma=1):
        ''' Initialize the attention module '''
        super().__init__()

        # ensure valid inputs
        assert embed_dim % num_heads == 0, \
            f'num_heads={num_heads} should evenly divide embed_dim={embed_dim}'

        # store off the scale and input params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.scale = scale
        self.kernel_size = kernel_size
        self.sigma = sigma
        # self.scale = self.projection_dim ** -0.5

        # Combine projections for multiple heads into a single linear layer for efficiency
        self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.sparsemax = Sparsemax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialization '''
        # Initialize using Xavier
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.input_weights, gain)
        nn.init.xavier_uniform_(self.output_projection.weight, gain)

    def project(self, inputs, index=0, chunks=1):
        ''' Produce a linear projection using the weights '''
        batch_size = inputs.shape[0]
        start = index * self.embed_dim
        end = start + chunks * self.embed_dim
        # import pdb; pdb.set_trace()
        projections = F.linear(inputs, self.input_weights[start:end]).chunk(chunks, dim=-1)

        output_projections = []
        for projection in projections:
            # transform projection to (BH x T x E)
            output_projections.append(
                projection.view(
                    batch_size,
                    -1,
                    self.num_heads,
                    self.projection_dim
                ).transpose(2, 1).contiguous().view(
                    batch_size * self.num_heads,
                    -1,
                    self.projection_dim
                )
            )

        return output_projections

    def attention(self, queries, keys, values):
        ''' Scaled dot product attention with optional masks '''
        logits = self.scale * torch.bmm(queries, keys.transpose(2, 1))

        num_patches = int(math.sqrt(logits.shape[-1]))
        bsz, num_groups, _ = logits.shape
        logits_reshape = logits.view(bsz, num_groups, num_patches, num_patches)
        logits_reshape_blurred = torchvision.transforms.functional.gaussian_blur(logits_reshape, self.kernel_size, self.sigma)
        attn_weights = F.softmax(logits_reshape_blurred.flatten(-2), dim=-1)
        return attn_weights

    def forward(self, queries, keys, values):
        ''' Forward pass of the attention '''
        # pylint:disable=unbalanced-tuple-unpacking
        if same_tensor(values, keys, queries):
            values, keys, queries = self.project(values, chunks=3)
        elif same_tensor(values, keys):
            values, keys = self.project(values, chunks=2)
            queries, = self.project(queries, 2)
        else:
            values, = self.project(values, 0)
            keys, = self.project(keys, 1)
            queries, = self.project(queries, 2)

        attn_weights = self.attention(queries, keys, values)
        return attn_weights


class GroupGenerateLayerBlur(nn.Module):
    def __init__(self, hidden_dim, num_heads, scale=1, kernel_size=1, sigma=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale = scale
        self.kernel_size = kernel_size
        self.sigma = sigma
       
        self.multihead_attns = nn.ModuleList([SparseMultiHeadedAttentionBlur(hidden_dim, 
                                                                             scale=scale,
                                                                             kernel_size=kernel_size,
                                                                             sigma=sigma) \
                                                for _ in range(num_heads)])

    def forward(self, query, key_value, epoch=0):
        """
            Use multiheaded attention to get mask
            Num_interpretable_heads = num_heads * seq_len
            Input: x (bsz, seq_len, hidden_dim)
                   if actual_x is not None, then use actual_x instead of x to compute attn_output
            Output: attn_outputs (bsz, num_heads * seq_len, seq_len, hidden_dim)
                    mask (bsz, num_heads, seq_len, seq_len)
        """

        if epoch == -1:
            epoch = self.num_heads
        
        head_i = epoch % self.num_heads
        if self.training:
            attn_weights = self.multihead_attns[head_i](query, key_value, key_value)
        else:
            attn_weights = []
            if epoch < self.num_heads:
                num_heads_use = head_i + 1
            else:
                num_heads_use = self.num_heads
            for head_j in range(num_heads_use):
                attn_weights_j = self.multihead_attns[head_j](query, key_value, key_value)
                attn_weights.append(attn_weights_j)
            attn_weights = torch.stack(attn_weights, dim=1)
        # import pdb; pdb.set_trace()
        # attn_weights = 
        return attn_weights

class GroupSelectLayerPower(nn.Module):
    def __init__(self, hidden_dim, scale=1, proj=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.proj = copy.deepcopy(proj)
        self.multihead_attn = SparseMultiHeadedAttention(hidden_dim, scale=scale)
        self.scale = scale
        
        embed_dim = self.multihead_attn.embed_dim
        identity_matrix = torch.eye(embed_dim)
        self.multihead_attn.input_weights.data = identity_matrix[None].expand(3, embed_dim, embed_dim).reshape(-1, embed_dim)

    def forward(self, query, key, value):
        """
            Use multiheaded attention to get mask
            Num_heads = num_heads * seq_len
            Input: x (bsz, seq_len, hidden_dim)
                   if actual_x is not None, then use actual_x instead of x to compute attn_output
            Output: attn_outputs (bsz, num_heads * seq_len, seq_len, hidden_dim)
                    mask (bsz, num_heads, seq_len, seq_len)
        """
        # x shape: (batch_size, sequence_length, hidden_dim)
        # x shape: (..., hidden_dim)
        if self.proj is not None:
            # query = self.proj(query) # query is the class weights, which doesn't need to use the last layer to encode. maybe it can be encoded with other ways
            # print('key a', key.shape)
            key = self.proj(key)[0]
            # print('key b', key.shape)
            
        epsilon = 1e-30
        # import pdb; pdb.set_trace()
        bsz, seq_len, hidden_dim = query.shape

        # Obtain attention weights
        attn_weights = self.multihead_attn(query, key, key)
        mask = attn_weights.transpose(-1, -2)

        # Apply attention weights on what to be attended
        new_shape = list(mask.shape) + [1] * (len(value.shape) - 3)
        attn_outputs = (value * mask.view(*new_shape)).sum(1)

        # attn_outputs of shape (bsz, num_masks, num_classes)
        return attn_outputs, mask
    
    
class SOPImageCls4(SOPImageCls):
    
    def __init__(self, 
                 config,
                 blackbox_model,
                 class_weights=None,
                 projection_layer=None,
                 k=0.2,
                 group_selector_proj=None
                 ):
        super().__init__(config,
                            blackbox_model,
                            class_weights,
                            projection_layer
                            )
        self.group_gen_blur_ks1 = config.group_gen_blur_ks1
        self.group_gen_blur_sigma1 = config.group_gen_blur_sigma1
        
        self.input_attn = GroupGenerateLayerBlur(hidden_dim=self.input_hidden_size,
                                             num_heads=self.num_heads,
                                             scale=self.group_gen_scale,
                                             kernel_size=config.group_gen_blur_ks1,
                                             sigma=config.group_gen_blur_sigma1
                                                )
        self.output_attn = GroupSelectLayerPower(hidden_dim=self.hidden_size,
                                            scale=self.group_sel_scale,
                                            proj=group_selector_proj)
        
        self.image_size = config.image_size if isinstance(config.image_size, 
                                                    collections.abc.Iterable) \
                                            else (config.image_size, config.image_size)
        self.num_channels = config.num_channels
        # attention args
        self.attn_patch_size = config.attn_patch_size
        
        self.init_grads()
        
        for name, param in self.projection.named_parameters():
            param.requires_grad = True
            
        self.k = k
            
        # self.projection_func = projection_func
        
    def forward(self, 
                inputs, 
                segs=None, 
                input_mask_weights=None,
                epoch=-1, 
                mask_batch_size=16,
                label=None,
                return_tuple=False,
                binary_threshold=-1):
        if epoch == -1:
            epoch = self.num_heads
        bsz, num_channel, img_dim1, img_dim2 = inputs.shape
        
        # Mask (Group) generation
        if input_mask_weights is None:
            grouped_inputs, input_mask_weights, c = self.group_generate(inputs, epoch, mask_batch_size, segs, binary_threshold)
        else:
            grouped_inputs = inputs.unsqueeze(1) * input_mask_weights.unsqueeze(2) # directly apply mask

        # Backbone model
        logits, pooler_outputs = self.run_backbone(grouped_inputs, mask_batch_size)
        if c is not None:
            logits = logits * c[:,:,None,None]

        # Mask (Group) selection & aggregation
        weighted_logits, output_mask_weights, logits, pooler_outputs = self.group_select(logits, pooler_outputs, img_dim1, img_dim2)
        
        if return_tuple:
            return self.get_results_tuple(weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label)
        else:
            return weighted_logits
    
    def group_generate(self, inputs, epoch, mask_batch_size, segs=None, binary_threshold=-1):
        bsz, num_channel, img_dim1, img_dim2 = inputs.shape
        c = None
        if segs is None:   # should be renamed "segments"
            num_patch = 14
            projected_inputs = self.projection(inputs)
            num_patches = projected_inputs.shape[-2:]
            projected_inputs = projected_inputs.flatten(2).transpose(1, 2)  # bsz, img_dim1 * img_dim2, num_channel
            projected_inputs = projected_inputs * self.projected_input_scale

            if self.num_masks_max != -1:
                projected_query = projected_inputs[:, :self.num_masks_max]
            else:
                projected_query = projected_inputs
            input_mask_weights_cand = self.input_attn(projected_query, projected_inputs, epoch=epoch)
            
            input_mask_weights_cand = input_mask_weights_cand.reshape(-1, num_patches[0]*num_patches[1])
            input_mask_weights_sort = input_mask_weights_cand.sort(-1)
            input_mask_weights_sort_values = input_mask_weights_sort.values.flip(-1)
            input_mask_weights_sort_indices = input_mask_weights_sort.indices.flip(-1)

            # get k scale
            topk = int(input_mask_weights_sort_values.shape[-1] * self.k)
            pos = input_mask_weights_sort_values[:,:topk]
            neg = input_mask_weights_sort_values[:,topk:]
            c = pos.sum(-1) - neg.sum(-1)
            c = c.view(bsz, -1)

            masks_all = torch.zeros_like(input_mask_weights_cand)
            masks_all[torch.arange(masks_all.size(0)).unsqueeze(1), input_mask_weights_sort_indices[:, :topk]] = 1
            masks_all = masks_all.view(bsz, -1, *num_patches)
            masks_all = torch.nn.functional.interpolate(masks_all, size=(img_dim1, img_dim2), mode='nearest')
            input_mask_weights_cand = masks_all
        else:
            # With/without masks are a bit different. Should we make them the same? Need to experiment.
            bsz, num_segs, img_dim1, img_dim2 = segs.shape
            seged_output_0 = inputs.unsqueeze(1) * segs.unsqueeze(2) # (bsz, num_masks, num_channel, img_dim1, img_dim2)
            _, interm_outputs = self.run_backbone(seged_output_0, mask_batch_size)
            
            interm_outputs = interm_outputs.view(bsz, -1, self.hidden_size)
            interm_outputs = interm_outputs * self.projected_input_scale
            segment_mask_weights = self.input_attn(interm_outputs, interm_outputs, epoch=epoch)
            segment_mask_weights = segment_mask_weights.reshape(bsz, -1, num_segs)
            
            new_masks =  segs.unsqueeze(1) * segment_mask_weights.unsqueeze(-1).unsqueeze(-1)
            input_mask_weights_cand = new_masks.sum(2)  # if one mask has it, then have it
            
            
        scale_factor = 1.0 / input_mask_weights_cand.reshape(bsz, -1, 
                                                        img_dim1 * img_dim2).max(dim=-1).values
        input_mask_weights_cand = input_mask_weights_cand * scale_factor.view(bsz, -1,1,1)
        
        # dropout
        dropout_mask = torch.zeros(bsz, input_mask_weights_cand.shape[1]).to(inputs.device)
        keep_group_idxs = torch.arange(self.num_masks_sample) * input_mask_weights_cand.shape[1] // self.num_masks_sample
        dropout_mask[:, keep_group_idxs] = 1
        if c is not None:
            c = c[dropout_mask.bool()].view(bsz, -1)
        
        input_mask_weights = input_mask_weights_cand[dropout_mask.bool()].clone()
        input_mask_weights = input_mask_weights.view(bsz, -1, img_dim1, img_dim2)

        masked_inputs = inputs.unsqueeze(1) * input_mask_weights.unsqueeze(2)
        return masked_inputs, input_mask_weights, c


class SOPText(SOP):
    def __init__(self, 
                 config,
                 blackbox_model,
                 class_weights=None,
                 projection_layer=None,
                 ):
        super().__init__(config,
                            blackbox_model,
                            class_weights
                            )

        if projection_layer is not None:
            self.projection = copy.deepcopy(projection_layer)
        else:
            self.init_projection()

        # Initialize the weights of the model
        self.init_grads()

    def init_projection(self):
        self.projection = nn.Linear(1, self.hidden_size)

    def forward(self, 
                inputs, 
                segs=None, 
                input_mask_weights=None,
                epoch=-1, 
                mask_batch_size=16,
                label=None,
                return_tuple=False,
                kwargs={},
                binary_threshold=-1):
        # import pdb; pdb.set_trace()
        if epoch == -1:
            epoch = self.num_heads
        bsz, seq_len = inputs.shape
        
        # Mask (Group) generation
        if input_mask_weights is None:
            grouped_inputs_embeds, input_mask_weights, grouped_kwargs = self.group_generate(inputs, epoch, mask_batch_size, 
                                                                                            segs, kwargs, binary_threshold)
            grouped_inputs = None
        else:
            grouped_inputs = inputs.unsqueeze(1) * input_mask_weights.unsqueeze(2) # directly apply mask

        # Backbone model
        logits, pooler_outputs = self.run_backbone(grouped_inputs, mask_batch_size, kwargs=grouped_kwargs)

        # Mask (Group) selection & aggregation
        weighted_logits, output_mask_weights, logits, pooler_outputs = self.group_select(logits, pooler_outputs, seq_len)

        if return_tuple:
            return self.get_results_tuple(weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label)
        else:
            return weighted_logits

    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label):
        raise NotImplementedError

    def run_backbone(self, masked_inputs=None, mask_batch_size=16, kwargs={}):  # TODO: Fix so that we don't need to know the input
        if masked_inputs is not None:
            bsz, num_masks, seq_len = masked_inputs.shape
            masked_inputs = masked_inputs.reshape(-1, seq_len)
            kwargs_flat = {k: v.reshape(-1, seq_len) for k, v in kwargs.items()}
        else:
            bsz, num_masks, seq_len, hidden_size = kwargs['inputs_embeds'].shape
            
            kwargs_flat = {k: v.reshape(-1, seq_len, hidden_size) if k == 'inputs_embeds' else v.reshape(-1, seq_len)
                           for k, v in kwargs.items()}
        logits = []
        pooler_outputs = []
        for i in range(0, bsz * num_masks, mask_batch_size):
            kwargs_i = {k: v[i:i+mask_batch_size] for k, v in kwargs_flat.items()}
            output_i = self.blackbox_model(
                masked_inputs[i:i+mask_batch_size] if masked_inputs is not None else None,
                **kwargs_i
            )
            pooler_i = output_i.pooler_output
            logits_i = output_i.logits
            logits.append(logits_i)
            pooler_outputs.append(pooler_i)

        logits = torch.cat(logits).view(bsz, num_masks, self.num_classes, -1)
        pooler_outputs = torch.cat(pooler_outputs).view(bsz, num_masks, self.hidden_size, -1)
        return logits, pooler_outputs
    
    def group_generate(self, inputs, epoch, mask_batch_size=16, segs=None, kwargs={}, binary_threshold=-1):
        bsz, seq_len = inputs.shape
        mask_embed = self.projection(torch.tensor([0]).int().to(inputs.device))
        projected_inputs = self.projection(inputs)
        
        if segs is None:   # word level
            projected_inputs = projected_inputs * self.projected_input_scale

            if self.num_masks_max != -1:
                input_dropout_idxs = torch.randperm(projected_inputs.shape[1]).to(inputs.device)
                if 'attention_mask' in kwargs:
                    attention_mask_mult = kwargs['attention_mask'] * input_dropout_idxs
                else:
                    attention_mask_mult = input_dropout_idxs
                input_dropout_idxs = torch.argsort(attention_mask_mult, dim=-1).flip(-1)[:, :self.num_masks_max]
                batch_indices = torch.arange(bsz).unsqueeze(1).repeat(1, input_dropout_idxs.shape[-1])
                selected_projected_inputs = projected_inputs[batch_indices, input_dropout_idxs]
                projected_query = selected_projected_inputs
            else:
                projected_query = projected_inputs
            input_mask_weights_cand = self.input_attn(projected_query, projected_inputs, epoch=epoch)
            input_mask_weights_cand = input_mask_weights_cand.squeeze(1)

            input_mask_weights_cand = torch.clip(input_mask_weights_cand, max=1.0)
        else: # sentence level
            # With/without masks are a bit different. Should we make them the same? Need to experiment.
            bsz, num_segs, seq_len = segs.shape

            seged_inputs_embeds = projected_inputs.unsqueeze(1) * segs.unsqueeze(-1) + \
                               mask_embed.view(1,1,1,-1) * (1 - segs.unsqueeze(-1))
            
            seged_kwargs = {}
            for k, v in kwargs.items():
                seged_kwargs[k] = v.unsqueeze(1).expand(segs.shape).reshape(-1, seq_len)
            seged_kwargs['inputs_embeds'] = seged_inputs_embeds

            # TODO: always have seg for the part after sep token
            _, interm_outputs = self.run_backbone(None, mask_batch_size, kwargs=seged_kwargs)
            
            interm_outputs = interm_outputs.view(bsz, -1, self.hidden_size)
            interm_outputs = interm_outputs * self.projected_input_scale
            segment_mask_weights = self.input_attn(interm_outputs, interm_outputs, epoch=epoch)
            segment_mask_weights = segment_mask_weights.reshape(bsz, -1, num_segs)
            
            new_masks =  segs.unsqueeze(1) * segment_mask_weights.unsqueeze(-1)
            # (bsz, num_new_masks, num_masks, seq_len)
            input_mask_weights_cand = new_masks.sum(2)  # if one mask has it, then have it
            # todo: Can we simplify the above to be dot product?
            
        scale_factor = 1.0 / input_mask_weights_cand.max(dim=-1).values
        input_mask_weights_cand = input_mask_weights_cand * scale_factor.view(bsz, -1,1)
        input_mask_weights_cand = torch.sigmoid(torch.log(input_mask_weights_cand / \
            self.group_gen_temp_beta + EPS) / self.group_gen_temp_alpha)
        if binary_threshold != -1 and not self.training: # if binary threshold is set, then use binary mask above the threshold only for testing
            input_mask_weights_cand = (input_mask_weights_cand > binary_threshold).float()

        # we are using iterative training
        # we will train some masks every epoch
        # the masks to train are selected by mod of epoch number
        # Dropout for training
        if self.training:
            dropout_idxs = torch.randperm(input_mask_weights_cand.shape[1])[:self.num_masks_sample]
            dropout_mask = torch.zeros(bsz, input_mask_weights_cand.shape[1]).to(inputs.device)
            dropout_mask[:,dropout_idxs] = 1
        else:
            dropout_mask = torch.ones(bsz, input_mask_weights_cand.shape[1]).to(inputs.device)
        
        input_mask_weights = input_mask_weights_cand[dropout_mask.bool()].clone()
        input_mask_weights = input_mask_weights.reshape(bsz, -1, seq_len)
        
        # Always add the second part of the sequence (in question answering, it would be the qa pair)
        if 'token_type_ids' in kwargs:
            input_mask_weights = input_mask_weights  + kwargs['token_type_ids'].unsqueeze(1)
        
        masked_inputs_embeds = projected_inputs.unsqueeze(1) * input_mask_weights.unsqueeze(-1) + \
                               mask_embed.view(1,1,1,-1) * (1 - input_mask_weights.unsqueeze(-1))
        
        masked_kwargs = {}
        for k, v in kwargs.items():
            masked_kwargs[k] = v.unsqueeze(1).expand(input_mask_weights.shape).reshape(-1, seq_len)
        masked_kwargs['inputs_embeds'] = masked_inputs_embeds
        
        return masked_inputs_embeds, input_mask_weights, masked_kwargs
    
    def group_select(self, logits, pooler_outputs, seq_len):
        raise NotImplementedError


class SOPTextCls(SOPText):
    def group_select(self, logits, pooler_outputs, seq_len):
        bsz, num_masks = logits.shape[:2]

        logits = logits.view(bsz, num_masks, self.num_classes)
        pooler_outputs = pooler_outputs.view(bsz, num_masks, self.hidden_size)

        query = self.class_weights.unsqueeze(0).expand(bsz, 
                                                    self.num_classes, 
                                                    self.hidden_size) #.to(logits.device)
        
        key = pooler_outputs
        # import pdb; pdb.set_trace()
        weighted_logits, output_mask_weights = self.output_attn(query, key, logits)
        # import pdb; pdb.set_trace()

        return weighted_logits, output_mask_weights, logits, pooler_outputs
    
    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label):
        # todo: debug for segmentation
        masks_aggr = None
        masks_aggr_pred_cls = None
        masks_max_pred_cls = None
        flat_masks = None

        if label is not None:
            predicted = label  # allow labels to be different
        else:
            _, predicted = torch.max(weighted_logits.data, -1)
        # import pdb; pdb.set_trace()
        # masks_mult = input_mask_weights.unsqueeze(2) * output_mask_weights.unsqueeze(-1) # bsz, n_masks, n_cls
        
        # masks_aggr = masks_mult.sum(1) # bsz, n_cls
        # masks_aggr_pred_cls = masks_aggr[range(bsz), predicted].unsqueeze(1)
        # max_mask_indices = output_mask_weights.max(2).values.max(1).indices
        # masks_max_pred_cls = masks_mult[range(bsz),max_mask_indices,predicted].unsqueeze(1)
            
        grouped_attributions = output_mask_weights * logits

        masks_mult_pred = input_mask_weights * output_mask_weights[range(len(predicted)),:,predicted,None]
        masks_aggr_pred_cls = masks_mult_pred.sum(1)
        max_mask_indices = output_mask_weights.max(2).values.max(1).indices
        masks_max_pred_cls = masks_mult_pred[range(bsz),max_mask_indices]

        # import pdb; pdb.set_trace()
        flat_masks = compress_masks_text(input_mask_weights, output_mask_weights[:,:,predicted])
        
        return AttributionOutputSOP(weighted_logits,
                                    logits,
                                    pooler_outputs,
                                    input_mask_weights,
                                    output_mask_weights,
                                    masks_aggr_pred_cls,
                                    masks_max_pred_cls,
                                    masks_aggr,
                                    flat_masks,
                                    grouped_attributions)