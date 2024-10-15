import random
import torch
import torch.nn as nn 

class RandomGroups(nn.Module): 
    def __init__(self, scaling = 1.5, distinct = 6, window_size=100, labels=None, past_values=None, past_time_features=None, past_observed_mask=None): 
        super(RandomGroups, self).__init__()
        self.scaling = scaling
        self.distinct = distinct
        self.window_size = window_size

    def forward(self, labels=None, past_values=None, past_time_features=None, past_observed_mask=None):       
        scaling, distinct, window_size = self.scaling, self.distinct, self.window_size
        max_group = scaling * distinct
        ngroups = random.randint(1, max_group)

        t = past_time_features[:,:,0]

        t_lower = t.min(dim=1).values
        t_upper = t.max(dim=1).values
        
        step_size = (t_upper - t_lower)/ngroups
        steps = torch.arange(0,ngroups,device=t.device)*(step_size.unsqueeze(1))
        time_start = t_lower[:,None] + steps
        time_end = torch.min(time_start + window_size, t_upper[:,None])
        
        return (time_start[:,:,None] <= t[:,None,:]) & (time_end[:,:,None] >= t[:,None,:])