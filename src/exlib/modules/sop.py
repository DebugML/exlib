# TODO: optimize

from __future__ import division
# import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from transformers import PreTrainedModel
import collections.abc
from functools import partial
# from torch.optim import Optimizer
# from torch.optim.lr_scheduler import LambdaLR
from transformers import PretrainedConfig, PreTrainedModel
import copy
import json
from collections import namedtuple
import os
# import math

import collections
import torchvision

# from sop_utils import *
from .libs.sop.utils import *
from .libs.sop.visualization import show_masks_weights, get_masks_used, show_masked_img, show_masks


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
                                   "group_attributions",
                                   "loss_scale"])


class GroupGenerateLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, scale=1, kernel_size=1, sigma=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale = scale
        self.kernel_size = kernel_size
        self.sigma = sigma
       
        self.multihead_attns = nn.ModuleList([MultiHeadedAttentionBlur(hidden_dim, 
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
        
        return attn_weights

    
class GroupSelectLayer(nn.Module):
    def __init__(self, hidden_dim, scale=1, proj=None, init_weights_identity=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.proj = copy.deepcopy(proj)
        self.multihead_attn = SparseMultiHeadedAttention(hidden_dim, scale=scale)
        self.scale = scale

        self.init_weights_identity = init_weights_identity
        if init_weights_identity:
            self.init_weights()

    def init_weights(self):
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
            key = self.proj(key)[0]
            
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


class SOPConfig(PretrainedConfig):
    def __init__(self,
                 json_file: str = None,
                 hidden_size = 512,
                 input_hidden_size = None,
                 num_labels = 2,
                 projected_input_scale = 1,
                 num_heads = 1,
                 num_masks_sample = 20,
                 num_masks_max = 200,
                 image_size=(224, 224),
                 num_channels = 3,
                 attn_patch_size = 16,
                 finetune_layers=[],
                 class_weight_layer = None,
                 group_gen_scale = 1,
                 group_sel_scale = 1,
                 group_gen_blur_ks1 = -1,
                 group_gen_blur_sigma1 = -1,
                 group_gen_blur_ks2 = -1,
                 group_gen_blur_sigma2 = -1,
                 freeze_projection = False,
                 **kwargs
                 ):
        # all the config from the json file will be in self.__dict__
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.input_hidden_size = input_hidden_size
        self.num_labels = num_labels
        self.projected_input_scale = projected_input_scale
        self.num_heads = num_heads
        self.num_masks_sample = num_masks_sample
        self.num_masks_max = num_masks_max
        self.image_size = image_size
        self.num_channels = num_channels
        self.attn_patch_size = attn_patch_size
        self.finetune_layers = finetune_layers
        self.class_weight_layer = class_weight_layer
        self.group_gen_scale = group_gen_scale
        self.group_sel_scale = group_sel_scale
        self.group_gen_blur_ks1 = group_gen_blur_ks1
        self.group_gen_blur_sigma1 = group_gen_blur_sigma1
        self.group_gen_blur_ks2 = group_gen_blur_ks2
        self.group_gen_blur_sigma2 = group_gen_blur_sigma2
        self.freeze_projection = freeze_projection
        
        self.json_file = json_file
        if json_file is not None:
            if os.path.isdir(json_file):
                json_file = os.path.join(json_file, 'config.json')
            if not os.path.exists(json_file):
                raise ValueError(f'config file {json_file} does not exist')
            self.update_from_json(json_file)
        
        if self.class_weight_layer is None:
            if self.finetune_layers is not None and len(self.finetune_layers) > 0:
                self.class_weight_layer = self.finetune_layers[0]
            else:
                # default
                self.class_weight_layer = 'classifier'
        
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
            'class_weight_layer',
            'group_gen_scale',
            'group_sel_scale',
            'group_gen_blur_ks1',
            'group_gen_blur_sigma1',
            'group_gen_blur_ks2',
            'group_gen_blur_sigma2',
            'freeze_projection',
        ]
        to_save = {k: v for k, v in self.__dict__.items() if k in attrs_save}
        with open(json_file, 'w') as f:
            json.dump(to_save, f, indent=4)


# class SOP(nn.Module):
class SOP(PreTrainedModel):
    config_class = SOPConfig

    def __init__(self, 
                 config=None,
                 backbone_model=None,
                 class_weights=None,
                 group_selector_proj=None,
                 k=0.2
                 ):
        super().__init__(config)
        # self.config = config
        self.hidden_size = self.config.hidden_size  # match black_box_model hidden_size
        self.input_hidden_size = self.config.input_hidden_size if (hasattr(self.config, 'input_hidden_size') is not None and hasattr(self.config, 'input_hidden_size') \
                                                                   and self.config.input_hidden_size is not None) else self.hidden_size
        self.num_classes = self.config.num_labels if hasattr(self.config, 'num_labels') is not None else 1  # 1 is for regression
        self.projected_input_scale = self.config.projected_input_scale if hasattr(self.config, 'projected_input_scale') else 1
        self.num_heads = self.config.num_heads
        self.num_masks_sample = self.config.num_masks_sample
        self.num_masks_max = self.config.num_masks_max
        self.class_weight_layer = self.config.class_weight_layer
        self.group_gen_scale = self.config.group_gen_scale if hasattr(self.config, 'group_gen_scale') else 1
        self.group_sel_scale = self.config.group_sel_scale if hasattr(self.config, 'group_sel_scale') else 1
        self.group_gen_blur_ks1 = self.config.group_gen_blur_ks1 if hasattr(self.config, 'group_gen_blur_ks1') else -1
        self.group_gen_blur_sigma1 = self.config.group_gen_blur_sigma1 if hasattr(self.config, 'group_gen_blur_sigma1') else -1
        self.group_gen_blur_ks2 = self.config.group_gen_blur_ks2 if hasattr(self.config, 'group_gen_blur_ks2') else -1
        self.group_gen_blur_sigma2 = self.config.group_gen_blur_sigma2 if hasattr(self.config, 'group_gen_blur_sigma2') else -1

        # blackbox model and finetune layers
        self.blackbox_model = backbone_model
        if class_weights is None:
            try:
                class_weights = get_chained_attr(backbone_model, self.config.class_weight_layer).weight
            except:
                class_weights = nn.Parameter(torch.Tensor(self.num_classes, self.hidden_size))
                gain = nn.init.calculate_gain('linear')
                nn.init.xavier_uniform_(class_weights, gain)
                print('class_weights is None and cannot be inferred from backbone_model. Initialized random matrices.')
                # raise ValueError('class_weights is None and cannot be inferred from backbone_model')
        
        self.class_weights = nn.Parameter(class_weights.clone())
        
        self.input_attn = GroupGenerateLayer(hidden_dim=self.input_hidden_size,
                                             num_heads=self.num_heads,
                                             scale=self.group_gen_scale,
                                             kernel_size=self.config.group_gen_blur_ks1,
                                             sigma=self.config.group_gen_blur_sigma1
                                                )
        self.output_attn = GroupSelectLayer(hidden_dim=self.hidden_size,
                                            scale=self.group_sel_scale,
                                            proj=group_selector_proj)
        self.k = k

    def init_grads(self):
        # Initialize the weights of the model
        if hasattr(self.config, 'freeze_projection') and (self.config.freeze_projection):
            for name, param in self.projection.named_parameters():
                param.requires_grad = False
            print('projection layer is frozen')
        else:
            print('projection layer is not frozen')

        for name, param in self.blackbox_model.named_parameters():
            param.requires_grad = False

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

    def load_checkpoint(self, checkpoint_path, strict=True):
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pth')
        self.load_state_dict(torch.load(checkpoint_path)['model'], strict=strict)
        print('Loaded model from checkpoint {}'.format(checkpoint_path))


class SOPImage(SOP):
    def __init__(self, 
                 config=None,
                 blackbox_model=None,
                 class_weights=None,
                 projection_layer=None,
                 group_selector_proj=None,
                 k=0.2
                 ):
        super().__init__(config,
                            blackbox_model,
                            class_weights,
                         group_selector_proj=group_selector_proj,
                         k=k
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

        self.init_grads()
        
        # allow finetuning the projection layer
        for name, param in self.projection.named_parameters():
            param.requires_grad = True


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
                binary_threshold=-1,
                separate_scale=False,
                deletion=False
                ):
        if epoch == -1:
            epoch = self.num_heads
        bsz, num_channel, img_dim1, img_dim2 = inputs.shape

        c = None
        
        # Mask (Group) generation
        if input_mask_weights is None:
            grouped_inputs, input_mask_weights, c = self.group_generate(inputs, epoch, mask_batch_size, segs, binary_threshold, deletion)
        else:
            grouped_inputs = inputs.unsqueeze(1) * input_mask_weights.unsqueeze(2) # directly apply mask

        # Backbone model
        logits, pooler_outputs = self.run_backbone(grouped_inputs, mask_batch_size)
        if c is not None and not separate_scale:
            logits = logits * c[:,:,None,None]

        # Mask (Group) selection & aggregation
        weighted_logits, output_mask_weights, logits, pooler_outputs = self.group_select(logits, pooler_outputs, img_dim1, img_dim2)
        
        if return_tuple:
            return self.get_results_tuple(
                weighted_logits, 
                logits, 
                pooler_outputs, 
                input_mask_weights, 
                output_mask_weights, 
                bsz, 
                label,
                c)
        else:
            return weighted_logits

    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, 
                        bsz, label, c):
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
    
    def group_generate(self, inputs, epoch, mask_batch_size, segs=None, binary_threshold=-1, deletion=False):
        bsz, num_channel, img_dim1, img_dim2 = inputs.shape
        c = None
        if segs is None:   # should be renamed "segments"
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
            input_mask_weights_sort_values = input_mask_weights_sort.values #.flip(-1)
            input_mask_weights_sort_indices = input_mask_weights_sort.indices #.flip(-1)
            if not deletion:
                input_mask_weights_sort_values = input_mask_weights_sort_values.flip(-1)
                input_mask_weights_sort_indices = input_mask_weights_sort_indices.flip(-1)

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
            raise NotImplementedError
            
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
                                                    self.hidden_size) 
        
        key = pooler_outputs
        weighted_logits, output_mask_weights = self.output_attn(query, key, logits)

        return weighted_logits, output_mask_weights, logits, pooler_outputs
    
    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, 
                          output_mask_weights, bsz, label, c):
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
        masks_mult_pred = input_mask_weights * grouped_attributions[range(len(predicted)),:,predicted,None,None]
        masks_aggr_pred_cls = masks_mult_pred.sum(1)[:,None,:,:]
        max_mask_indices = grouped_attributions.max(2).values.max(1).indices
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
                                    grouped_attributions,
                                    c)

SOPImageCls4 = SOPImageCls

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
    
    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, 
                          bsz, label, c):
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
        return AttributionOutputSOP(weighted_logits,
                                    logits,
                                    pooler_outputs,
                                    input_mask_weights,
                                    output_mask_weights,
                                    masks_aggr_pred_cls,
                                    masks_max_pred_cls,
                                    masks_aggr,
                                    flat_masks,
                                    grouped_attributions,
                                    c)


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
                                    grouped_attributions,
                                    None)