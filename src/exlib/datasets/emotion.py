import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn as nn
import sentence_transformers

import sys
sys.path.append("../src")
import exlib
# Baselines
from exlib.features.text.text_chunk import text_chunk
from exlib.utils.emotion_helper import project_points_onto_axes, load_emotions


MODEL_REPO = "BrachioLab/roberta-base-go_emotions"
DATASET_REPO = "BrachioLab/emotion"
TOKENIZER_REPO = "roberta-base"


def load_data():
    hf_dataset = load_dataset(DATASET_REPO)
    return hf_dataset


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


#classifier for go emotions dataset
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(MODEL_REPO)
        self.classifier = nn.Linear(768, 28)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

class Metric(nn.Module): 
    def __init__(self, model_name:str="all-mpnet-base-v2"): 
        super(Metric, self).__init__()
        self.model = sentence_transformers.SentenceTransformer(model_name)
        points = self.define_circumplex()
        self.x1 = points[0]
        self.x2 = points[1]
        self.y1 = points[3]
        self.y2 = points[2]

    def define_circumplex(self):
        # Circumplex labels to emotions
        label_to_emotions = {
            # Positive valence
            "PV": ["Happy", "Pleased", "Delighted", "Excited", "Satisfied"],
            # Negative valence
            "NV": ["Miserable", "Frustrated", "Sad", "Depressed", "Afraid"],
            # High arousal
            "HA": ["Astonished", "Alarmed", "Angry", "Afraid", "Excited"],
            # Low arousal
            "LA": ["Tired", "Sleepy", "Calm", "Satisfied", "Depressed"],
        }
        axis_points = []
        for k, v in label_to_emotions.items():
            emotion_embeddings = self.model.encode(np.array(v))
            axis_points.append(np.mean(emotion_embeddings, axis=0))
        return axis_points
    
    def distance_from_circumplex(self, embeddings):
        projection = project_points_onto_axes(embeddings, self.x1, self.x2, self.y1, self.y2)
        x_projections = projection[0]
        y_projections = projection[1]
        distances = []
        for x, y in zip(x_projections, y_projections):
            distances.append(np.abs(np.sqrt(x**2 + y**2)-1))
        return 1/np.mean(distances)

    def mean_pairwise_dist(self, embeddings):
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

    # input: list of words
    def calculate_group_alignment(self, groups:list, language:str="english"):
        alignments = []
        for group in groups:
            embeddings = self.model.encode(group)
            circumplex_dist = self.distance_from_circumplex(embeddings)
            if(len(embeddings) == 1): 
                # alignments.append(circumplex_dist)
                final_dist = np.exp(-circumplex_dist)
                alignments.append(final_dist)
            else:
                mean_dist = self.mean_pairwise_dist(embeddings)
                # print(circumplex_dist, mean_dist)
                # combined_dist = circumplex_dist*mean_dist
                combined_dist = np.exp(-circumplex_dist*mean_dist)
                alignments.append(combined_dist)
        return alignments
        
    def forward(self, zp, x=None, y=None, z=None, reduce=True, **kwargs): 
        pass


def get_emotion_scores(baselines = ['word', 'phrase', 'sentence']):
    dataset = EmotionDataset("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier()
    model.to(device)
    model.eval()

    metric = Metric()
    torch.manual_seed(1234)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    all_baselines_scores = {}
    for baseline in baselines:
        print(f"---- {baseline} Level Groups ----")
        
        baseline_scores = []
        for i, batch in enumerate(tqdm(dataloader)):
            word_lists = batch['word_list']
            word_lists = list(map(list, zip(*word_lists)))
            processed_word_lists = []
            for word_list in word_lists:
                processed_word_lists.append([word for word in word_list if word != ''])
            
            for word_list in processed_word_lists:
                groups = text_chunk(word_list, baseline)
                # print(groups)
                alignments = torch.tensor(metric.calculate_group_alignment(groups))
                score = alignments.mean()
                baseline_scores.append(score)
            
            # if i > 1:
            #     break
                
        # print(baseline_scores)    
        baseline_scores = torch.stack(baseline_scores)
        all_baselines_scores[baseline] = baseline_scores
    
    # print(all_baselines_scores)
    return all_baselines_scores
