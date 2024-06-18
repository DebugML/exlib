import torch
import torch.nn as nn 

class BaselineGroups(nn.Module): 
    def __init__(self, ngroups=10, window_size=100, labels=None, past_values=None, past_time_features=None, past_observed_mask=None): 
        super(BaselineGroups, self).__init__()
        self.ngroups = ngroups
        self.window_size = window_size

    def forward(self, labels=None, past_values=None, past_time_features=None, past_observed_mask=None): 
        # x.size = (n_examples, 300, 4)
        t = past_time_features[:,:,0]
        ngroups, window_size = self.ngroups, self.window_size

        t_lower = t.min(dim=1).values
        t_upper = t.max(dim=1).values
        
        step_size = (t_upper - t_lower)/ngroups
        steps = torch.arange(0,ngroups,device=t.device)*(step_size.unsqueeze(1))
        time_start = t_lower[:,None] + steps
        time_end = torch.min(time_start + window_size, t_upper[:,None])
        
        return (time_start[:,:,None] <= t[:,None,:]) & (time_end[:,:,None] >= t[:,None,:])