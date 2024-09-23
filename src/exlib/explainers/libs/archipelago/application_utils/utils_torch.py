import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.optim as optim


class ModelWrapperTorch:
    def __init__(self, model, device, input_type="image"):
        self.device = device
        self.model = model.to(device)
        self.input_type = input_type

    def __call__(self, X):
        if self.input_type == "text":
            X = torch.LongTensor(X).to(self.device)
            preds = self.model(X)[0].data.cpu().numpy()

        elif self.input_type == "time_series":
            X = torch.FloatTensor(X).to(self.device)
            X = X.squeeze(3)
            past_values = X[:, 0:2].permute(0, 2, 1)
            past_time_features = X[:, 2:4].permute(0, 2, 1)
            past_observed_mask = X[:, 4:6]
            preds = self.model(past_values = past_values, past_time_features = past_time_features, past_observed_mask = past_observed_mask).logits.data.cpu().numpy()
            preds = preds.squeeze(1)
                                  
        else:
            X = torch.FloatTensor(X).to(self.device)
            if self.input_type == "image":
                X = X.permute(0, 3, 1, 2)
            preds = self.model(X).data.cpu().numpy()
        return preds
