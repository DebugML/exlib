import copy
import torch
import torch.nn.functional as F
import numpy as np
from .common import *

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std



def pre_processing(obs, torch_device):
    # rescale imagenet, we do mornalization in the network, instead of preprocessing
    # mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    # std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    obs = obs / 255
    # obs = (obs - mean) / std
    obs = np.transpose(obs, (2, 0, 1))
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)
    # if cuda:
    #     torch_device = torch.device('cuda:0')
    # else:
    #     torch_device = torch.device('cpu')
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device)
    return obs_tensor

#%%
def fgsm_step(image, epsilon, data_grad_adv, data_grad_lab):
    # generate the perturbed image based on steepest descent
    grad_lab_norm = torch.norm(data_grad_lab,p=2)
    delta = epsilon * data_grad_adv.sign()

    # + delta because we are ascending
    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)
    delta = perturbed_rect - image
    delta = - data_grad_lab * delta
    return perturbed_rect, delta
    # return perturbed_image, delta

def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    c_delta = 0 # cumulative delta
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # if attack is successful, then break
        if pred.item() == targeted.item():
            break
        # select the false class label
        output = F.softmax(output, dim=1)
        loss = output[0,targeted.item()]

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        loss_lab = output[0, init_pred.item()]
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(image, epsilon, data_grad_adv, data_grad_lab)
        c_delta += delta
    
    return c_delta, perturbed_image


class AgiImageCls(FeatureAttrMethod):
    def __init__(self, model, max_iter=15, topk=15, epsilon=0.05):
        super().__init__(model)
        self.max_iter = max_iter
        self.topk = topk
        self.epsilon = epsilon

    def forward(self, x, t, **kwargs):
        assert len(x) == len(t)

        attrs = []
        with torch.enable_grad():
            for i in range(len(x)):

                data = pre_processing(x[i:i+1].cpu().numpy()[0].transpose(1,2,0), x.device)
                data = data.to(x.device)

                output = self.model(data)

                if t is None:
                    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                else:
                    init_pred = t[i:i+1]
                num_classes = output.size(-1)
                selected_ids = range(0,num_classes - 1,int(num_classes/self.topk))

                top_ids = selected_ids # only for predefined ids
                # initialize the step_grad towards all target false classes
                step_grad = 0 
                # num_class = 1000 # number of total classes
                for l in top_ids:

                    targeted = torch.tensor([l]).to(x.device) 
                    if targeted.item() == init_pred.item():
                        continue # we don't want to attack to the predicted class.

                    delta, perturbed_image = pgd_step(x[i:i+1], self.epsilon, self.model, init_pred, targeted, self.max_iter)
                    step_grad += delta

                # adv_ex = step_grad.squeeze().detach().cpu().numpy() # / topk
                attrs.append(step_grad)
        return FeatureAttrOutput(torch.cat(attrs, 0).to(x.device), {})