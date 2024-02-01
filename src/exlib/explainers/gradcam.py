# TODO: fix gradcam itself to make it generalize

import torch
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from .common import *
from copy import deepcopy
from .libs.pytorch_grad_cam.base_cam_text import BaseCAMText


class WrappedModelGradCAM(torch.nn.Module):
    def __init__(self, model): 
        super(WrappedModelGradCAM, self).__init__()
        self.model = model
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.model(x)


class GradCAMImageCls(FeatureAttrMethod):
    def __init__(self, model, target_layers, reshape_transform=None):
        
        model = WrappedModelGradCAM(model)

        super().__init__(model)
        
        self.target_layers = target_layers
        if reshape_transform is None:
            def reshape_transform(tensor, height=14, width=14):
                result = tensor[:, 1 :  , :].reshape(tensor.size(0),
                    height, width, tensor.size(2))

                # Bring the channels to the first dimension,
                # like in CNNs.
                result = result.transpose(2, 3).transpose(1, 2)
                return result
            
            
        with torch.enable_grad():
            self.grad_cam = GradCAM(model=model, target_layers=self.target_layers,
                                    reshape_transform=reshape_transform,
                                    use_cuda=True if torch.cuda.is_available() else False)

    def forward(self, X, label=None, target_func=ClassifierOutputSoftmaxTarget):
        grad_cam_results = []
        for i in range(len(label)):
            with torch.enable_grad():
                grad_cam_result = self.grad_cam(input_tensor=X[i:i+1], 
                                                targets=[target_func(label[i:i+1])])
                grad_cam_result = torch.tensor(grad_cam_result)
                grad_cam_results.append(grad_cam_result)
        grad_cam_results = torch.stack(grad_cam_results)

        return FeatureAttrOutput(grad_cam_results, grad_cam_result)
    

class GradCAMTextCls(FeatureAttrMethod):
    def __init__(self, model, target_layers, reshape_transform=None):
        model = WrappedModelGradCAM(model)

        super().__init__(model)
        
        self.target_layers = target_layers
        with torch.enable_grad():
            self.grad_cam = GradCAMText(model=model, target_layers=self.target_layers,
                                    reshape_transform=reshape_transform,
                                    use_cuda=True if torch.cuda.is_available() else False)
 
    def forward(self, X, label=None, target_func=ClassifierOutputSoftmaxTarget):
        grad_cam_results = []
        for i in range(len(label)):
            with torch.enable_grad():
                grad_cam_result = self.grad_cam(input_tensor=X[i:i+1], 
                                                targets=[target_func(label[i:i+1])])
                # import pdb; pdb.set_trace()
                grad_cam_result = torch.tensor(grad_cam_result)
                grad_cam_results.append(grad_cam_result)
        grad_cam_results = torch.cat(grad_cam_results)

        return FeatureAttrOutput(grad_cam_results.squeeze(1), {})