import torch
import torch.nn as nn
import numpy as np
# from exlib.utils.supernova_helper import *

class KMeansTorch:
    def __init__(self, n_clusters=7, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        indices = torch.randperm(X.size(0))[:self.n_clusters]
        centroids = X[indices].clone()
        
        for _ in range(self.max_iter):
            distances = torch.cdist(X, centroids)
            labels = torch.argmin(distances, dim=1)
            new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(self.n_clusters)])
            if torch.all(centroids == new_centroids):
                break
            centroids = new_centroids
        self.centroids = centroids
        return labels

    def predict(self, X):
        distances = torch.cdist(X, self.centroids)
        labels = torch.argmin(distances, dim=1)
        return labels

class ClusteringGroups(nn.Module): 
    def __init__(self, nclusters=7):
        super(ClusteringGroups, self).__init__()
        self.nclusters = nclusters

    def forward(self, labels=None, past_values=None, past_time_features=None, past_observed_mask=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        nclusters = self.nclusters
        t, wl, flux, err = past_time_features[:,:,0].to(device), past_time_features[:,:,1].to(device), past_values[:,:,0].to(device), past_values[:,:,1].to(device)
        datas = torch.stack([t, wl, flux, err], dim=1)
        unique_wl = torch.Tensor([3670.69, 4826.85, 6223.24, 7545.98, 8590.9, 9710.28]).to(device)
        
        bool_tensor = wl[:,None,None,:] == unique_wl[:,None].to(device)
        float_tensor = datas.unsqueeze(2)
        datas_by_wl = torch.where(bool_tensor, float_tensor, torch.tensor(0.0, device=device))
        wl = datas_by_wl[:, 3, :, :]
        time_series_data = wl.reshape(-1, 300).to(device)
        num = len(wl) * len(wl[0])
        clusters = np.zeros((num, nclusters, 300))

        kmeans = KMeansTorch(n_clusters=nclusters)
        for i in range(num):
            cluster_labels = kmeans.fit(time_series_data[i].reshape(-1, 1).to(device))
            for j in range(300):
                cluster_idx = cluster_labels[j]
                clusters[i][cluster_idx][j] = 1

        clusters = torch.tensor(clusters).to(device)
        pred_groups = clusters.bool()

        return pred_groups
