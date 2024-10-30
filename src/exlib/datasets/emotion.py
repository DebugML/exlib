import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import datasets as hfds
import torch.nn as nn
try:
    import sentence_transformers
except Exception as e:
    print(f'sentence_transformers has errors: {e}')

from pathlib import Path

import os
import sys
sys.path.append("../src")
import exlib
# Baselines
from exlib.features.text import *

from .emotion_helper import project_points_onto_axes, load_emotions
from .common import BaseFixScore

# MODEL_REPO = "BrachioLab/roberta-base-go_emotions"
MODEL_REPO = "SamLowe/roberta-base-go_emotions"
DATASET_REPO = "BrachioLab/emotion"
TOKENIZER_REPO = "roberta-base"


def load_data():
    hf_dataset = hfds.load_dataset(DATASET_REPO)
    return hf_dataset


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(MODEL_REPO)
    model.to(device)
    return model


#go emotions dataset
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        dataset = hfds.load_dataset(DATASET_REPO)[split]
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

#classifier for go emotions dataset
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        torch.manual_seed(1234)
        print(MODEL_REPO)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)

    def forward(self, input_ids, attention_mask= None):
        outputs = self.model(input_ids, attention_mask)
        return outputs
    

class EmotionFixScore(BaseFixScore): 
    def __init__(self, model_name:str="all-mpnet-base-v2"): 
        super().__init__()
        self.model = sentence_transformers.SentenceTransformer(model_name)
        points = self.define_circumplex()
        self.x1 = points[0]
        self.x2 = points[1]
        self.y1 = points[3]
        self.y2 = points[2]

    def define_circumplex(self):
        emotions = load_emotions()
        axis_labels = ["NV", "PV", "HA", "LA"]
        axis_points = []
        for label in axis_labels:
            emotion_words = emotions[label]
            emotion_embeddings = self.model.encode(emotion_words)
            axis_points.append(np.mean(emotion_embeddings, axis=0))
        return axis_points
    
    def distance_from_circumplex(self, embeddings): # signal
        projection = project_points_onto_axes(embeddings, self.x1, self.x2, self.y1, self.y2)
        x_projections = projection[0]
        y_projections = projection[1]
        distances = []
        for x, y in zip(x_projections, y_projections):                
            distances.append(np.abs(np.sqrt(x**2 + y**2)-1))
        return 1/np.mean(distances)

    def mean_pairwise_dist(self, embeddings): # relatedness
        projection = project_points_onto_axes(embeddings, self.x1, self.x2, self.y1, self.y2)
        distances = []
        x_coords = projection[0]
        y_coords = projection[1]
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                x_dist = x_coords[i] - x_coords[j]
                y_dist = y_coords[i] - y_coords[j]
                distances.append(np.sqrt(x_dist**2 + y_dist**2))
        return 1/np.mean(distances)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def tanh(self, x):
        return 2*self.sigmoid(x) - 1

    # input: list of words
    def calculate_group_alignment(self, groups:list, language:str="english"):
        alignments = []
        for group in groups:
            embeddings = self.model.encode(group)
            circumplex_dist = self.distance_from_circumplex(embeddings)

            if(len(embeddings) == 1): 
                alignments.append(circumplex_dist)
            else:
                mean_dist = self.mean_pairwise_dist(embeddings)
                combined_dist = circumplex_dist*mean_dist
                alignments.append(combined_dist)
#         alignments = [self.tanh(np.exp(-a)) for a in alignments]
        alignments = [self.tanh(np.exp(-a)) for a in alignments]
        return alignments
    
    def forward(self, groups_pred:list, x:list, language="english", reduce=True): # original_data is processed_word_list
        #create groups
        group_masks = groups_pred
        original_data = x
        groups = []
        for i in range(len(group_masks)):
            mask = group_masks[i]
            group = [original_data[j] for j in range(len(mask)) if mask[j] == 1 and original_data[j] != '']
            if group != []:
                groups.append(group)
        #print(groups)
        scores = self.calculate_group_alignment(groups, language)
        if reduce:
            if len(scores) == 0:
                return 0.0
            else:
                return np.mean(scores)
        else:
            return scores


r"""
Some code for running the FIX score on different baselines.
"""

_all_emotion_baselines = [
    'identity', 
    'random', 
    'word', 
    'phrase', 
    'sentence', 
    'clustering', 
    'archipelago'
]        
        
def get_emotion_scores(
    baselines = _all_emotion_baselines,
    utterances_path = 'utterances/emotion_test.pt',
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    distinct = 4,
    scaling = 1.5,
    batch_size = 4,
    N = None,
):
    torch.manual_seed(1234)
    dataset = EmotionDataset("test")

    if N is not None:
        dataset = Subset(dataset, torch.randperm(len(dataset))[:N].tolist())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = EmotionClassifier().to(device).eval()
    metric = EmotionFixScore()

    all_baselines_scores = {}
    for baseline in baselines:
        #print(f"---- {baseline} Level Groups ----")
        
        baseline_scores = []
        if baseline == 'clustering':
#             utterances_path = 'utterances/emotion_test.pt'
            if os.path.exists(utterances_path):
                utterances = torch.load(utterances_path)
            else:
                utterances = [' '.join(dataset[i]['word_list']) for i in range(len(dataset))]

                # Make the utterances directory if it does not exist
                ut_path = Path(utterances_path)
                ut_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save(utterances, str(ut_path))
            groups = ClusteringGroups(utterances, distinct=distinct, scaling=scaling)
        
        for i, batch in enumerate(tqdm(dataloader)):

            word_lists = batch['word_list']
            word_lists = list(map(list, zip(*word_lists)))

            processed_word_lists = []
            for word_list in word_lists:
                processed_word_lists.append([word for word in word_list if word != ''])
            
            if baseline == 'archipelago': # get masks by batch
                backbone_model = model
                groups = ArchipelagoGroups(backbone_model, distinct=distinct, scaling=scaling)
                all_batch_masks = groups(batch)
               
            
            for example in range(len(word_lists)):
                if baseline == 'identity':
                    groups = IdentityGroups()
                    masks = groups(word_lists[example])
                elif baseline == 'random':
                    groups = RandomGroups(distinct=distinct, scaling=scaling)
                    masks = groups(word_lists[example])
                if baseline == 'word':
                    groups = WordGroups(distinct=distinct, scaling=scaling)
                    masks = groups(word_lists[example])
                elif baseline == 'phrase':
                    groups = PhraseGroups(distinct=distinct, scaling=scaling)
                    masks = groups(word_lists[example])
                elif baseline == 'sentence':
                    groups = SentenceGroups(distinct=distinct, scaling=scaling)
                    masks = groups(word_lists[example])
                elif baseline == 'clustering':
                    masks = groups(word_lists[example])
                elif baseline == 'archipelago': # get score for each example with the already generated masks
                    masks = all_batch_masks[example]
                #print(word_lists[example])
                score = metric(masks, word_lists[example])

                baseline_scores.append(score)

        baseline_scores = torch.tensor(baseline_scores)
        all_baselines_scores[baseline] = baseline_scores
    
    return all_baselines_scores


def preprocess_emotion(batch):
    x = batch['word_list']
    X = {'x': x[0]} # politeness can only do batch size of 1
    metric_inputs = {'x': x[0]}
    return X, metric_inputs