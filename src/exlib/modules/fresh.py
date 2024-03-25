from __future__ import division
import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
import collections.abc
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import copy
# from pathlib import Path
# import sys
# print(Path(__file__).parents[0])
# print(Path(__file__).parents[1])
# path_root = Path(__file__).parents[1]
# print(path_root)
# sys.path.append(str(path_root))
from collections import namedtuple

AttributionOutputFresh = namedtuple("AttributionOutputFresh", 
                                  ["logits",
                                   "attributions"])

class TopKAttentionLayer(nn.Module):
    def __init__(self, k=0.2):
        super().__init__()
        assert(k > 0)
        self.k = k

    def forward(self, attn, attn_mask=None):
        """
            Get topk attention
            Input: attn (bsz, seq_len)
                   attn_mask (bsz, seq_len)  (1 and 0)
            Output: mask
        """
        device = attn.device
        if attn_mask is None:
            attn_mask = torch.ones_like(attn)
        
        if 0 < self.k < 1:
            k_values = (torch.sum(attn_mask, dim=-1) * self.k).int()
            if k_values.min() == 0:
                k_values[k_values == 0] = 1
        else:
            k_values = self.k * torch.ones(attn.size(0), dtype=torch.int)

        # Rank the attention values
        sorted_indices = torch.argsort(attn, descending=True, dim=-1)
        
        # Create a mask of the same shape as attn and fill it with zeros
        mask = torch.zeros_like(attn, dtype=torch.float).to(device)
        
        # Use arange to create an index tensor 
        idx = torch.arange(attn.size(1)).unsqueeze(0).expand_as(attn).to(device)
        
        # Create a boolean mask for top-k values
        top_k_mask = idx < k_values.unsqueeze(-1)
        
        # Use the boolean mask to get the top-k indices for each row
        top_k_indices = torch.gather(sorted_indices, 1, top_k_mask.long().cumsum(dim=-1) - 1)
        
        # Set the top-k positions in the mask tensor to 1
        mask.scatter_(1, top_k_indices, 1)

        return mask * attn_mask  # Ensuring we keep original zeros from attn_mask
    

class FRESH(PreTrainedModel):
    def __init__(self, 
                 config,
                 blackbox_model,
                 model_type='image',
                 return_tuple=False,
                 postprocess_attn=None,
                 postprocess_logits=None,
                 projection_layer=None,
                 rationale_len=0.2
                 ):
        if config is not None:
            super().__init__(config)
        else:
            super().__init__()
        self.config = config
        self.model_type = model_type
        self.return_tuple = return_tuple
        self.postprocess_attn = postprocess_attn
        self.postprocess_logits = postprocess_logits
        self.rationale_len = rationale_len

        if model_type == 'image':
            self.image_size = config.image_size if isinstance(config.image_size, 
                                                        collections.abc.Iterable) \
                                                else (config.image_size, config.image_size)
            self.num_channels = config.num_channels
        else:  # text
            self.image_size = None
            self.num_channels = None

        self.num_classes = config.num_labels if config.num_labels is not None else 1  # 1 is for regression
        
        # self.pooler = pooler

        # attention args
        if model_type == 'image':
            self.attn_patch_size = config.attn_patch_size
            if hasattr(config, 'attn_stride_size') and \
                config.attn_stride_size is not None:
                self.attn_stride_size = config.attn_stride_size
            else:
                self.attn_stride_size = config.attn_patch_size
        else:
            self.attn_patch_size = None
            self.attn_stride_size = None

        # blackbox model and finetune layers
        self.blackbox_model = blackbox_model
        self.finetune_layers = config.finetune_layers # e.g. ['classifier', 'fc']
        if hasattr(config, 'finetune_layers_idxs'):
            self.finetune_layers_idxs = config.finetune_layers_idxs
        else:
            self.finetune_layers_idxs = None

        # attention - new parts
        # input
        self.input_attn = TopKAttentionLayer(k=rationale_len)

        # if projection_layer is not None:
        #     self.projection = copy.deepcopy(projection_layer)
        if model_type == 'image':
            self.projection_up = nn.ConvTranspose2d(1, 
                                                    1, 
                                                    kernel_size=self.attn_patch_size, 
                                                    stride=self.attn_stride_size)  # make each patch a vec
            self.projection_up.weight = nn.Parameter(torch.ones_like(self.projection_up.weight))
            self.projection_up.bias = torch.nn.Parameter(torch.zeros_like(self.projection_up.bias))
            self.projection_up.weight.requires_grad = False
            self.projection_up.bias.requires_grad = False
        # else:  # text
        #     self.projection = nn.Linear(1, self.proj_hid_size)

        # Initialize the weights of the model
        self.init_weights()
        self.blackbox_model = blackbox_model
        self.blackbox_model_pred = copy.deepcopy(blackbox_model)
        if self.finetune_layers_idxs is None:
            self.class_weights = copy.deepcopy(getattr(self.blackbox_model, self.finetune_layers[0]).weight)
            # Freeze the frozen module
            for name, param in self.blackbox_model.named_parameters():
                if sum([ft_layer in name for ft_layer in self.finetune_layers]) == 0: # the name doesn't match any finetune layers
                    param.requires_grad = False
        else:
            self.class_weights = copy.deepcopy(getattr(self.blackbox_model, self.finetune_layers[0])[self.finetune_layers_idxs[0]].weight)
            # Freeze the frozen module
            for name, param in self.blackbox_model.named_parameters():
                if sum([f'{self.finetune_layers[i]}.{self.finetune_layers_idxs[i]}' in name for i in range(len(self.finetune_layers))]) == 0: # the name doesn't match any finetune layers
                    param.requires_grad = False
        

    def forward(self, inputs, token_type_ids=None, attention_mask=None, pad_token_id=0):
        if self.model_type == 'image':
            bsz, num_channel, img_dim1, img_dim2 = inputs.shape
        else:
            bsz, seq_len = inputs.shape

        with torch.no_grad():
            if self.model_type == 'image':
                outputs = self.blackbox_model(
                    inputs,
                    output_attentions=True,
                    return_dict=True
                )
                attn = self.postprocess_attn(outputs)
                input_mask_weights = self.input_attn(attn, attn_mask=attention_mask)
                # print('input_mask_weights', input_mask_weights.shape)
                # import pdb
                # pdb.set_trace()
                # todo: project attn to pixels
                num_patches = ((self.image_size[0] - self.attn_patch_size) \
                            // self.attn_stride_size + 1, 
                            (self.image_size[1] - self.attn_patch_size) \
                                // self.attn_stride_size + 1)
                input_mask_weights = input_mask_weights[:,1:].reshape(-1, 
                                                                1, 
                                                                num_patches[0], 
                                                                num_patches[1])
                input_mask_weights = self.projection_up(input_mask_weights, 
                                                        output_size=torch.Size([input_mask_weights.shape[0], 
                                                                                1, 
                                                                                img_dim1, 
                                                                                img_dim2]))
                input_mask_weights = input_mask_weights.view(bsz, 
                                                            -1, 
                                                            img_dim1, 
                                                            img_dim2)
                input_mask_weights = torch.clip(input_mask_weights, max=1.0)
                # import pdb
                # pdb.set_trace()

                masked_inputs = input_mask_weights * inputs
            else:
                if token_type_ids is None:
                    outputs = self.blackbox_model(
                        inputs,
                        attention_mask=attention_mask,
                        output_attentions=True,
                        return_dict=True
                    )
                else:
                    outputs = self.blackbox_model(
                        inputs,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        output_attentions=True,
                        return_dict=True
                    )
                attn = self.postprocess_attn(outputs)
                input_mask_weights = self.input_attn(attn, attn_mask=attention_mask)
                
                if token_type_ids is None:
                    to_keep = input_mask_weights * attention_mask
                else:
                    to_keep = (token_type_ids[0] == 1) * attention_mask  # query
                    document_attn = (token_type_ids[0] == 0) * attention_mask * input_mask_weights
                    to_keep = to_keep + document_attn
                # compress TODO
                # Step 1: Masking and Compacting
                masked_inputs = [inputs[i][to_keep[i].bool()] for i in range(to_keep.shape[0])]
                if token_type_ids is not None:
                    masked_token_type_ids = [token_type_ids[i][to_keep[i].bool()] for i in range(to_keep.shape[0])]
                else:
                    masked_token_type_ids = None
                masked_attention_mask = [attention_mask[i][to_keep[i].bool()] for i in range(to_keep.shape[0])]

                # Step 2: Padding
                max_len = inputs.shape[-1]
                padded_inputs = torch.stack([torch.cat([m, torch.zeros(max_len - m.shape[0], device=inputs.device)]) \
                                             for m in masked_inputs])
                if token_type_ids is not None:
                    padded_token_type_ids = torch.stack([torch.cat([m, torch.zeros(max_len - m.shape[0], device=inputs.device)]) \
                                             for m in masked_token_type_ids])
                else:
                    padded_token_type_ids = None
                padded_attention_mask = torch.stack([torch.cat([m, torch.zeros(max_len - m.shape[0], device=inputs.device)]) \
                                             for m in masked_attention_mask])
        
        if self.model_type == 'image':
            outputs = self.blackbox_model_pred(masked_inputs)
        else:  # text
            if padded_token_type_ids is None:
                outputs = self.blackbox_model_pred(padded_inputs.int(),
                                              attention_mask=padded_attention_mask.int())
            else:
                outputs = self.blackbox_model_pred(padded_inputs.int(),
                                            token_type_ids=padded_token_type_ids.int(),
                                            attention_mask=padded_attention_mask.int())
        
        if self.return_tuple: 
            return AttributionOutputFresh(self.postprocess_logits(outputs),
                                          input_mask_weights)
        else:
            return self.postprocess_logits(outputs)
