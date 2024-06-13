import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import tqdm
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn as nn
import sentence_transformers


DATASET_REPO = "go_emotions"
MODEL_REPO = "shreyahavaldar/roberta-base-go_emotions"
TOKENIZER_REPO = "roberta-base"

def load_data():
    hf_dataset = load_dataset(DATASET_REPO)
    return hf_dataset

def load_model():
    model = AutoModel.from_pretrained(MODEL_REPO)
    model.to(device)
    return model

#go emotions dataset
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        dataset = load_dataset(DATASET_REPO)[split]        
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
        self.max_len = max([len(text.split()) for text in dataset['text']])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['labels'][0]
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        word_list = text.split()
        for i in range(len(word_list), self.max_len):
            word_list.append('')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label),
            'word_list': word_list
        }
