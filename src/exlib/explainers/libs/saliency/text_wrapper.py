import torch.nn as nn


class ExtraDimModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x_embed):
        # print('x_embed', x_embed.shape)
        if x_embed.ndim == 4:
            x_embed = x_embed.squeeze(1)
        return self.model(inputs_embeds=x_embed)