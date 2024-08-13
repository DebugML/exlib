import torch
import torch.nn as nn
from collections import namedtuple

from .libs.xdnn.imagenet.models import AlexNet, XAlexNet, vgg16, xvgg16, fixup_resnet50, xfixup_resnet50
from collections import OrderedDict
import torchvision.transforms as transforms



AttributionOutputXDNN = namedtuple("AttributionOutputXDNN", [
    "logits",
    "attributions"
    ])


class XDNN(nn.Module):
    def __init__(self,
                 model_name,
                 model_path=None):
        super().__init__()

        self.model_name = model_name

        self.type_mapping = {
            'alexnet': AlexNet,
            'xalexnet': XAlexNet,
            'vgg16': vgg16,
            'xvgg16': xvgg16,
            'fixup_resnet50': fixup_resnet50,
            'xfixup_resnet50': xfixup_resnet50
        }

        self.model = self.type_mapping[model_name]()

        if model_path is not None:
            checkpoint = torch.load(model_path)
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:7] == 'module.':
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)

        # prepare data
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.unnormalize = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                            std = [ 1., 1., 1. ])])

    def forward(self, x, t=None):
        with torch.enable_grad():
            images = self.normalize(x).detach()
            # compute output
            

            images = images.to(x.device)
            images.requires_grad = True # make sure this is set to True
            

            outputs = self.model(images)

            if t is None:
                t = outputs.argmax(dim=1)
            target = t

            target = target.to(x.device)
            

            # compute attribution
            target_outputs = torch.gather(outputs, 1, target.unsqueeze(-1))
            gradients = torch.autograd.grad(torch.unbind(target_outputs), images, create_graph=False, allow_unused=True)[0]
            attributions = gradients * images

            return AttributionOutputXDNN(outputs, attributions)