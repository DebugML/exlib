from exlib.explainers.common import FeatureAttrMethod, FeatureAttrOutput
import torch.nn as nn
import torch

class AttnImageCls(FeatureAttrMethod):
    def __init__(self, model, img_dim=224, num_patches=14, patch_size=16, 
                 attn_kwargs={'output_attentions': True, 'return_dict': True}):
        super().__init__(model)
        self.attn_kwargs = attn_kwargs
        self.img_dim = img_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection_up = nn.ConvTranspose2d(1, 1, kernel_size=patch_size, 
                                                stride=patch_size)
        self.projection_up.weight = nn.Parameter(torch.ones_like(self.projection_up.weight))
        self.projection_up.bias = torch.nn.Parameter(torch.zeros_like(self.projection_up.bias))
        self.projection_up.weight.requires_grad = False
        self.projection_up.bias.requires_grad = False
        
    def forward(self, x, labels=None, kwargs={}):
        self.model.eval()
        with torch.no_grad():
            original_outputs = self.model(x, **self.attn_kwargs, **kwargs)
        original_logits = original_outputs.logits
        attn = original_outputs.attentions[-1].mean(dim=1)[:,0,1:]

        attn = attn.reshape(-1, 1, self.num_patches, self.num_patches)
        attn = self.projection_up(attn, output_size=torch.Size([attn.shape[0], 1, 
                                                                self.img_dim, 
                                                                self.img_dim]))
        return FeatureAttrOutput(attn, {})