import torch
import torch.nn as nn
from collections import namedtuple
import bagnets.pytorchnet
from bagnets.utils import plot_heatmap, generate_heatmap_pytorch
import torchvision.transforms as transforms


AttributionOutputBagNet = namedtuple("AttributionOutputBagNet", [
    "logits",
    "attributions"
    ])


class BagNet(nn.Module):
    def __init__(self,
                 model_name='bagnet33',
                 pretrained=True,
                 patchsize=None):
        super().__init__()

        self.model_name = model_name

        self.type_mapping = {
            'bagnet9': (bagnets.pytorchnet.bagnet9, 9),
            'bagnet17': (bagnets.pytorchnet.bagnet17, 17),
            'bagnet33': (bagnets.pytorchnet.bagnet33, 33)
        }

        self.model = self.type_mapping[model_name][0](pretrained=pretrained)
        if patchsize is None:
            self.patchsize = self.type_mapping[model_name][1]
        else:
            self.patchsize = patchsize

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

    def forward(self, x, t=None):
        x = self.normalize(x)
        outputs = self.model(x)
        if t is None:
            t = outputs.argmax(dim=1)
        attrs = []
        for i in range(len(x)):
            heatmap = generate_heatmap_pytorch(self.model, x[i:i+1].cpu(), t[i:i+1].cpu(), 
                        self.patchsize)
            attrs.append(torch.tensor(heatmap)[None])

        return AttributionOutputBagNet(outputs, torch.stack(attrs, 0))