import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import sentence_transformers

def load_data():
    hf_dataset = load_dataset(DATASET_REPO)
    return hf_dataset

def load_model():
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_REPO)
    model.to(device)
    return model

class PolitenessDataset(torch.utils.data.Dataset):
    def __init__(self, split, language="english"):
        dataset = load_dataset(DATASET_REPO)[split]
        dataset = dataset.filter(lambda x: x["language"] == language)
        dataset = dataset.rename_column("politeness", "label")
        self.dataset = dataset
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(TOKENIZER_REPO)
        self.max_len = max([len(text.split()) for text in dataset['Utterance']])


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset["Utterance"][idx]
        label = self.dataset["label"][idx]
        encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        word_list = text.split()
        for i in range(len(word_list), self.max_len):
            word_list.append('')
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label),
            'word_list': word_list
        }
