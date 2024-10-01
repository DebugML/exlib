"""
preprocess non-GP-interpolated data for use with learnable Fourier PE
"""

from functools import lru_cache
from functools import partial

from typing import Optional, Iterable

import torch
from torch.utils.data import DataLoader
from transformers import PretrainedConfig, set_seed
from transformers.models.informer.modeling_informer import InformerMeanScaler, InformerStdScaler, InformerNOPScaler
from datasets import concatenate_datasets

import numpy as np
import pandas as pd
import pdb
from collections import Counter

from typing import List, Optional, Tuple, Union
import torch

from transformers import InformerPreTrainedModel, InformerModel, InformerForPrediction, InformerConfig

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

from datasets import load_dataset
from transformers import InformerConfig

import torch
from torch.optim import AdamW

from tqdm.auto import tqdm
import argparse
import yaml
from pathlib import Path
import shutil
from datetime import datetime
    
from transformers.models.informer.modeling_informer import InformerConvLayer, InformerEncoder, InformerEncoderLayer, InformerDecoder,  weighted_average, nll
from transformers.modeling_outputs import SequenceClassifierOutput, Seq2SeqTSModelOutput, BaseModelOutput, Seq2SeqTSPredictionOutput, BaseModelOutputWithPastAndCrossAttentions, MaskedLMOutput
from transformers.time_series_utils import StudentTOutput, NormalOutput, NegativeBinomialOutput

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, functional as F
import pdb
import numpy as np
import random


@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def normalize_all(example, field_name):
    # normalize to [0,1] overall (min-max normalization to get rid of negative values)
    values = example[field_name]
    example[field_name] = (values - np.min(values)) / (np.max(values) - np.min(values))
    return example

def normalize_by_channel(example, field_name):
    # normalize to [0,1] by channel (min-max normalization to get rid of negative values)
    for row in range(len(example[field_name])):
        min_value = np.min(example[field_name][row][example[field_name][row] != 0])
        row_values = example[field_name][row]
        example[field_name][row] = (row_values - min_value) / (np.max(row_values) - min_value)
    return example

def create_attention_mask(example):
    # create attention mask to ignore padding
    example["attention_mask"] = np.zeros_like(example["transposed_target"])
    example["attention_mask"][:, example['transposed_times_wv'][0] != 0] = 1 # mask if time value is 0 (padding)
    return example

def mask(example, mask_fraction=0.5, mask_value=0):
    # mask out mask_fraction % of values in the target
    indices_to_replace = np.random.choice(len(example['transposed_target'][0]), int(len(example['transposed_target'][0]) * mask_fraction), replace=False)
    # replace 80% with mask_value, 10% with random value, 10% with original value (Devlin et al. 2018)
    indices_to_mask = np.random.choice(indices_to_replace, int(len(indices_to_replace) * 0.8), replace=False)
    remaining_indices = np.setdiff1d(indices_to_replace, indices_to_mask)
    indices_to_replace_with_random = np.random.choice(remaining_indices, int(len(remaining_indices) * 0.5), replace=False)
    # label for calculating loss: original value for masked, 0 for unmasked (don't want to calculate loss for unmasked)
    unmasked_indices = np.setdiff1d(range(len(example['transposed_target'][0])), indices_to_replace)
    example["mask_label"] = example["transposed_target"]
    example['mask_label'][:, unmasked_indices] = 0

    example['transposed_target'][:, indices_to_mask] = mask_value
    random_indices = np.random.choice(unmasked_indices, len(indices_to_replace_with_random), replace=False)
    example['transposed_target'][:, indices_to_replace_with_random] = example['transposed_target'][:, random_indices]

    return example

def masked_data_collator(mask_probability, cols_to_keep, data):
    batch = {}
    # defaultdict(partial(np.ndarray, 0))
    for key in data[0].keys():
        batch_key = key if key not in ['values', 'observed_mask', 'time_features'] else f"past_{key}"
        if batch_key not in cols_to_keep:
            continue
        batch[batch_key] = torch.stack([torch.tensor(example[key]) for example in data]) if key != 'objid' else [example[key] for example in data]

    labels = batch['past_values'][:, 0, :].clone() # only take flux values, should be [batch_size, 1, seq_len]

    masked_indices = torch.bernoulli(torch.full(labels.shape, mask_probability)).bool().squeeze()
    labels[~masked_indices] = 0  # We only compute loss on masked tokens

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().squeeze() & masked_indices

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().squeeze() & masked_indices & ~indices_replaced

    indices_replaced = torch.tile(torch.unsqueeze(indices_replaced, 1), (1, 2, 1))
    indices_random = torch.tile(torch.unsqueeze(indices_random, 1), (1, 2, 1))
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    batch['past_values'][indices_replaced] = 0
    batch['past_values'][indices_random] = torch.rand(batch['past_values'][indices_random].shape)
    batch['past_values'] = torch.transpose(batch['past_values'], 1, 2)
    batch['past_time_features'] = torch.transpose(batch['past_time_features'], 1, 2)

    if 'mask_label' in cols_to_keep:
        batch['mask_label'] = labels

    return batch

def transform_raw_data_example(example):
    # was 300 x 2, need to be 2 x 300 (first dim is channel)
    example['transposed_target'] = np.array(example['target']).T
    example['transposed_times_wv'] = np.array(example['times_wv']).T
    # divide by max value to constrain to [0,1]
    example = create_attention_mask(example)
    #example = normalize_by_channel(example, "transposed_times_wv")
    #example = normalize_all(example, "transposed_target") # normalize flux, flux_err by max overall
    return example

def transform_raw_data_example_model(example):
    # was 300 x 2, need to be 2 x 300 (first dim is channel)
    example['transposed_target'] = np.array(example['target']).T
    example['transposed_times_wv'] = np.array(example['times_wv']).T
    # divide by max value to constrain to [0,1]
    example = create_attention_mask(example)
    example = normalize_by_channel(example, "transposed_times_wv")
    example = normalize_all(example, "transposed_target") # normalize flux, flux_err by max overall
    return example


def transform_raw_data(dataset):
    # normalize time, data; create attention mask
    dataset = dataset.map(transform_raw_data_example)
    print(f"original dataset size: {len(dataset)}")
    # filter out nans
    dataset = dataset.filter(lambda example: not np.isnan(example['transposed_target']).any() and not np.isnan(example['transposed_times_wv']).any())
    print(f"remove nans dataset size: {len(dataset)}")
    # have to swap out these field names because can't change dataset field shapes in place
    dataset = dataset.remove_columns(["target", "times_wv"])
    dataset = dataset.rename_column("transposed_target", "target")
    dataset = dataset.rename_column("transposed_times_wv", "times_wv")

    # remove/rename fields
    name_mapping = {
                "times_wv": "time_features",
                "target": "values",
                "attention_mask": "observed_mask",
            }

    dataset = dataset.rename_columns(name_mapping)
    dataset = dataset.with_format('np')

    return dataset

def transform_raw_data_model(dataset):
    # normalize time, data; create attention mask
    dataset = dataset.map(transform_raw_data_example_model)
    print(f"original dataset size: {len(dataset)}")
    # filter out nans
    dataset = dataset.filter(lambda example: not np.isnan(example['transposed_target']).any() and not np.isnan(example['transposed_times_wv']).any())
    print(f"remove nans dataset size: {len(dataset)}")
    # have to swap out these field names because can't change dataset field shapes in place
    dataset = dataset.remove_columns(["target", "times_wv"])
    dataset = dataset.rename_column("transposed_target", "target")
    dataset = dataset.rename_column("transposed_times_wv", "times_wv")

    # remove/rename fields
    name_mapping = {
                "times_wv": "time_features",
                "target": "values",
                "attention_mask": "observed_mask",
            }

    dataset = dataset.rename_columns(name_mapping)
    dataset = dataset.with_format('np')

    return dataset


def create_test_dataloader_raw(
    #config: PretrainedConfig,
    dataset,
    batch_size: int,
    seed: Optional[int] = 42,
    add_objid: Optional[bool] = False,
    compute_loss: Optional[bool] = False,
):
    config = dataset.config
    dataset = dataset.dataset
    set_seed(seed)
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")
        if "redshift" in dataset.column_names:
            dataset = dataset.rename_column("redshift", "static_real_features")

    if config.has_labels:
        PREDICTION_INPUT_NAMES.append("labels")
        dataset = dataset.rename_column("label", "labels")

    if add_objid:
        PREDICTION_INPUT_NAMES.append("objid")

    if compute_loss:
        PREDICTION_INPUT_NAMES += [
            "future_values",
            "future_observed_mask",
        ]

    transformed_data = transform_raw_data(dataset).flatten_indices()
    #transformed_data = transformed_data.shuffle(seed=seed)  # TODO add seed to args
    mask_probability = 0. if config.has_labels else config.mask_probability # don't mask for fine-tuning
    return DataLoader(
        transformed_data,
        batch_size=batch_size,
        # sampler=sampler,
        num_workers=0,
        collate_fn=partial(masked_data_collator, mask_probability, PREDICTION_INPUT_NAMES)
    )

def create_test_dataloader(
    #config: PretrainedConfig,
    dataset,
    batch_size: int,
    seed: Optional[int] = 42,
    add_objid: Optional[bool] = False,
    compute_loss: Optional[bool] = False,
):
    config = dataset.config
    dataset = dataset.dataset
    set_seed(seed)
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")
        if "redshift" in dataset.column_names:
            dataset = dataset.rename_column("redshift", "static_real_features")

    if config.has_labels:
        PREDICTION_INPUT_NAMES.append("labels")
        dataset = dataset.rename_column("label", "labels")

    if add_objid:
        PREDICTION_INPUT_NAMES.append("objid")

    if compute_loss:
        PREDICTION_INPUT_NAMES += [
            "future_values",
            "future_observed_mask",
        ]

    transformed_data = transform_raw_data_model(dataset).flatten_indices()
    #transformed_data = transformed_data.shuffle(seed=seed)  # TODO add seed to args
    mask_probability = 0. if config.has_labels else config.mask_probability # don't mask for fine-tuning
    return DataLoader(
        transformed_data,
        batch_size=batch_size,
        # sampler=sampler,
        num_workers=0,
        collate_fn=partial(masked_data_collator, mask_probability, PREDICTION_INPUT_NAMES)
    )

def create_network_inputs(
    config: PretrainedConfig,
    past_values: torch.Tensor,
    past_time_features: torch.Tensor,
    static_categorical_features: Optional[torch.Tensor] = None,
    static_real_features: Optional[torch.Tensor] = None,
    past_observed_mask: Optional[torch.Tensor] = None,
    future_values: Optional[torch.Tensor] = None,
    future_time_features: Optional[torch.Tensor] = None,
):
    if config.scaling == "mean" or config.scaling is True:
        scaler = InformerMeanScaler(dim=1, keepdim=True)
    elif config.scaling == "std":
        scaler = InformerStdScaler(dim=1, keepdim=True)
    else:
        scaler = InformerNOPScaler(config)

    past_length = config.context_length + max(config.lags_sequence)
    # time feature
    time_feat = (
        torch.cat(
            (
                past_time_features[:, past_length - config.context_length :, ...],
                future_time_features,
            ),
            dim=1,
        )
        if future_values is not None
        else past_time_features[:, past_length - config.context_length :, ...]
    )

    # target
    if past_observed_mask is None:
        past_observed_mask = torch.ones_like(past_values)

    context = past_values[:, -config.context_length :]
    observed_context = past_observed_mask[:, -config.context_length :]
    _, loc, scale = scaler(context, observed_context)

    inputs = (
        (torch.cat((past_values, future_values), dim=1) - loc) / scale
        if future_values is not None
        else (past_values - loc) / scale
    )

    # static features
    log_abs_loc = loc.abs().log1p() if config.input_size == 1 else loc.squeeze(1).abs().log1p()
    log_scale = scale.log() if config.input_size == 1 else scale.squeeze(1).log()
    static_feat = torch.cat((log_abs_loc, log_scale), dim=1)

    if static_real_features is not None:
        static_feat = torch.cat((static_real_features.unsqueeze(1), static_feat), dim=1)
    if static_categorical_features is not None:
        embedded_cat = embedder(static_categorical_features)
        static_feat = torch.cat((embedded_cat, static_feat), dim=1)
    expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)

    # all features
    features = torch.cat((expanded_static_feat, time_feat), dim=-1)

    # lagged features
    subsequences_length = (
        config.context_length + config.prediction_length
        if future_values is not None
        else config.context_length
    )
    lagged_sequence = get_lagged_subsequences(config=config, sequence=inputs, subsequences_length=subsequences_length)
    lags_shape = lagged_sequence.shape
    reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

    if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
        raise ValueError(
            f"input length {reshaped_lagged_sequence.shape[1]} and time feature lengths {time_feat.shape[1]} does not match"
        )

    # transformer inputs
    transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

    return transformer_inputs, loc, scale, static_feat

def get_lagged_subsequences(
    config: PretrainedConfig, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
) -> torch.Tensor:
    sequence_length = sequence.shape[1]
    indices = [lag - shift for lag in config.lags_sequence]

    if max(indices) + subsequences_length > sequence_length:
        raise ValueError(
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )

    lagged_values = []
    for lag_index in indices:
        begin_index = -lag_index - subsequences_length
        end_index = -lag_index if lag_index > 0 else None
        lagged_values.append(sequence[:, begin_index:end_index, ...])
    return torch.stack(lagged_values, dim=-1)

class MultiDimFourierPE(nn.Module):
    """
    Implements Li et al. 2021 learnable multi-dimensional fourier positional encoding
    """

    def __init__(self, fourier_dim: int, hidden_dim: int, model_dim: int) -> None:
        super().__init__()
        self.fourier_dim = fourier_dim # e.g. 384
        self.hidden_dim = hidden_dim # e.g. 32
        self.embed_dim = model_dim # e.g. 768

        # initialize layers here to avoid device incompatibilities
        self.fourier_to_hidden = nn.Linear(self.fourier_dim, self.hidden_dim)
        self.hidden_to_embed = nn.Linear(self.hidden_dim, self.embed_dim)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: bsz x seq_len x num_dimensions
            - num_dimensions should be 2: [:,0] = time, [:,1] = central wavelength of band
            - both already scaled to be between 0 and 1

        Returns:
            pos_encoding: bsz x seq_len x d_model
        """
        bsz, seq_len, num_dimensions = pos.size()
        out = torch.zeros(bsz, seq_len, self.fourier_dim, device=pos.device, dtype=pos.dtype)
        sentinel = self.fourier_dim // 2 if self.fourier_dim % 2 == 0 else (self.fourier_dim // 2) + 1

        fourier_coeff = nn.Linear(num_dimensions, sentinel, bias=False, device=pos.device) # map num_dimensions to half of the fourier feature dimension
        out[:, :, 0:sentinel] = torch.sin(fourier_coeff(pos))
        out[:, :, sentinel:] = torch.cos(fourier_coeff(pos))
        out *= (1 / np.sqrt(self.fourier_dim)) # scale by sqrt of fourier dimension

        out = self.fourier_to_hidden(out) # map to hidden dimension
        out = nn.GELU()(out)
        out = self.hidden_to_embed(out) # map to embedding dimension

        return out

class InformerEncoderFourierPE(InformerEncoder):
    """
    Informer encoder consisting of *config.encoder_layers* self attention layers with distillation layers. Each
    attention layer is an [`InformerEncoderLayer`]. This implementation includes a custom positional encoding, CustomPE.

    Args:
        config: InformerConfig
    """

    def __init__(self, config: InformerConfig):
        super().__init__(config)
        print("Using Fourier PE")
        fourier_dim = config.fourier_dim if hasattr(config, "fourier_dim") else 384
        hidden_dim = config.PE_hidden_dim if hasattr(config, "PE_hidden_dim") else 32
        self.embed_positions = MultiDimFourierPE(fourier_dim, hidden_dim, config.d_model)

        # Initialize weights and apply final processing
        self.post_init()

    # copied from InformerEncoder
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # inputs_embeds[:,:,0]:
            # flux     XXXX
            # flux_err XXXX
            # mean     XXXX from scaling, not sure why they have 2 rows each
            # mean     XXXX
            # std      XXXX
            # std      XXXX
            # time     XXXX
            # wv       XXXX
        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(inputs_embeds[:,:,-2:]) # last two rows of features dimension are time and central wavelength

        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, (encoder_layer, conv_layer) in enumerate(zip(self.layers, self.conv_layers)):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                    if conv_layer is not None:
                        output = torch.utils.checkpoint.checkpoint(conv_layer, layer_outputs[0])
                        layer_outputs = (output,) + layer_outputs[1:]
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )
                    if conv_layer is not None:
                        output = conv_layer(layer_outputs[0])
                        layer_outputs = (output,) + layer_outputs[1:]

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class InformerDecoderFourierPE(InformerDecoder):
    """
    Informer decoder consisting of *config.decoder_layers* layers. Each layer is a [`InformerDecoderLayer`]

    Args:
        config: InformerConfig
    """

    def __init__(self, config: InformerConfig):
        super().__init__(config)
        print("Using Fourier PE")
        fourier_dim = config.fourier_dim if hasattr(config, "fourier_dim") else 384
        hidden_dim = config.PE_hidden_dim if hasattr(config, "PE_hidden_dim") else 32
        self.embed_positions = MultiDimFourierPE(fourier_dim, hidden_dim, config.d_model)

        # Initialize weights and apply final processing
        self.post_init()

    # copied from InformerDecoder
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            attention_mask (torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
               Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = inputs_embeds.size()[:-1]

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(inputs_embeds[:,:,-2:]) # last two rows of features dimension are time and central wavelength
        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class InformerFourierPEModel(InformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = InformerEncoderFourierPE(config)
        self.decoder = InformerDecoderFourierPE(config)
        # self.decoder = InformerDecoderFourierPE(config)

        # Initialize weights and apply final processing
        self.post_init()

class MaskedInformerFourierPE(InformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config.distil = False # no convolutions

        self.encoder = InformerEncoderFourierPE(self.config)
        self.decoder = MaskedInformerDecoder(self.config)

        # Initialize weights and apply final processing
        self.post_init()

    #TODO: write forward(), add MaskedInformerDecoder class
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None, # -100 for padding, true value for masked tokens
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqTSModelOutput, Tuple]: #TODO: not seq2seq output type

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_inputs, loc, scale, static_feat = self.create_network_inputs(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
        )

        outputs = self.encoder(
            inputs_embeds=transformer_inputs,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        prediction_scores = self.decoder(outputs[0]) # 1D array of predicted values (TODO: does it need to go through softmax? Bert doesn't)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = MSELoss(reduction='sum')  # 0 = unmasked token, mask from loss
            mask = labels.view(-1) != 0 # flattened
            #TODO: maybe multiply with attention mask to avoid computing loss on padding
            masked_predictions = prediction_scores.view(-1) * mask # multiply by 0 where mask array is 0, 1 where mask array is nonzero
            masked_lm_loss = loss_fct(masked_predictions, labels.view(-1)) # MSE(0,0) = 0 for masked predictions
            masked_lm_loss = masked_lm_loss / mask.sum()

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class MaskedInformerDecoder(InformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.decoder = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, values):
        return self.decoder(values)

class InformerForSequenceClassification(InformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        print(f"num labels: {self.num_labels}")
        self.config = config

        if config.fourier_pe and config.mask:
            config.distil = False
            self.encoder = InformerEncoderFourierPE(config)
        elif config.fourier_pe:
            self.model = InformerFourierPEModel(config)
        else:
            self.model = InformerModel(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        print(f"classifier dropout: {classifier_dropout}")
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.mask:
            #TODO: should encapsulate this preprocessing + encoder into its own class
            transformer_inputs, loc, scale, static_feat = create_network_inputs(
                config=self.config,
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                static_categorical_features=static_categorical_features,
                static_real_features=static_real_features,
            )

            outputs = self.encoder(
                inputs_embeds=transformer_inputs,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            decoder_output = outputs.last_hidden_state
            pooled_output = torch.mean(decoder_output, dim=1, keepdim=True) # average over time dimension

        else:
            outputs = self.model(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_values=future_values,
                future_time_features=future_time_features,
                static_categorical_features=static_categorical_features,
                static_real_features=static_real_features,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                use_cache=use_cache,
                return_dict=return_dict,
            )

            decoder_output = outputs.last_hidden_state

            pooled_output = torch.mean(decoder_output, dim=1, keepdim=True) # average over time dimension

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1 and self.config.regression:
                loss_fn = MSELoss()
            elif self.config.num_labels == 1:
                labels = labels.float()
                loss_fn = BCEWithLogitsLoss()
            else:
                loss_fn = CrossEntropyLoss(weight=weights) if weights is not None else CrossEntropyLoss()
            loss = loss_fn(logits.squeeze(), labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.last_hidden_state,
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--save_model", type=str)
    parser.add_argument("--load_model", type=str)
    parser.add_argument("--load_checkpoint", type=str)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--mask_probability", default=0.6, type=float)

    return parser.parse_args()

def get_dataset(data_dir, data_subset_file=None, force_redownload=False):
    kwargs = {}
    if data_subset_file is not None:
        with open(data_subset_file) as f:
            data_subset = [x.strip() for x in f.readlines()]
            print(f"using data subset: {data_subset}")

            kwargs["data_files"] = {'train': data_subset}
    if force_redownload:
        kwargs["download_mode"] = "force_redownload"

    dataset = load_dataset(data_dir, **kwargs)
    print(f"loading dataset {'from file ' if data_subset_file is not None else ''}with {len(dataset['train'])} examples")

    return dataset

def save_model(model, optimizer, output_dir):
    print(f"Saving model to {output_dir}")
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    torch_model_dir = Path(output_dir) / "torch_model"
    hf_model_dir = Path(output_dir) / "hf_model"

    print(f"overwriting torch model at {torch_model_dir}")
    if torch_model_dir.exists():
        shutil.rmtree(torch_model_dir)
    torch_model_dir.mkdir(parents=True)

    print(f"overwriting hf model at {hf_model_dir}")
    if hf_model_dir.exists():
        shutil.rmtree(hf_model_dir)
    hf_model_dir.mkdir(parents=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, torch_model_dir / "model.pt")

    model.save_pretrained(hf_model_dir)

def prepare_model_input(batch, device, config, mask):
    model_inputs = {
            "past_time_features": batch['past_time_features'].to(device),
            "past_values": batch["past_values"].to(device),
            "past_observed_mask": batch["past_observed_mask"].to(device),
    }
    if config.num_static_categorical_features > 0:
        model_inputs["static_categorical_features"] = batch["static_categorical_features"].to(device)
    if config.num_static_real_features > 0:
        model_inputs["static_real_features"] = batch["static_real_features"].to(device)
    if not mask:
        model_inputs["future_time_features"] = batch["future_time_features"].to(device)
        model_inputs["future_observed_mask"] = batch["future_observed_mask"].to(device)
        model_inputs["future_values"] = batch["future_values"].to(device)
    else:
        model_inputs["labels"] = batch["mask_label"].to(device)

    return model_inputs

def setup_model_config(args, config):
    # model config computes certain properties, can't config.update these
    model_config = InformerConfig(
        input_size=2,
        prediction_length=0,
        context_length=300,
        lags_sequence=[0],
        num_time_features=2, #wavelength + time
        num_static_real_features=0,

        # informer params:
        dropout=config['dropout_rate'],
        encoder_layers=config['num_encoder_layers'],
        decoder_layers=config['num_decoder_layers'],
        d_model=config['d_model'],
        scaling=config['scaling'],
        has_labels=False,
        mask=True,
        mask_probability=args.mask_probability,
    )

    addl_config = {}
    # additional encoder/decoder hyperparams:
    if 'encoder_attention_heads' in config:
        addl_config['encoder_attention_heads'] = config['encoder_attention_heads']
    if 'decoder_attention_heads' in config:
        addl_config['decoder_attention_heads'] = config['decoder_attention_heads']
    if 'encoder_ffn_dim' in config:
        addl_config['encoder_ffn_dim'] = config['encoder_ffn_dim']
    if 'decoder_ffn_dim' in config:
        addl_config['decoder_ffn_dim'] = config['decoder_ffn_dim']
    # additional hyperparams for learnable fourier PE:
    if 'fourier_dim' in config:
        addl_config['fourier_dim'] = config['fourier_dim']
    if 'PE_hidden_dim' in config:
        addl_config['PE_hidden_dim'] = config['PE_hidden_dim']

    model_config.update(addl_config)

    return model_config