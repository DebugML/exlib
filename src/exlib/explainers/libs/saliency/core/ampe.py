import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from .attack_methods import DI,gkern
from torch.autograd import Variable as V
from .dct import *
from scipy import stats as st
features = None


def hook_feature(module, input, output):
    global features
    features = output


def image_transform(x):
    return x


def get_NAA_loss(adv_feature, base_feature, weights):
    gamma = 1.0
    attribution = (adv_feature - base_feature) * weights
    blank = torch.zeros_like(attribution)
    positive = torch.where(attribution >= 0, attribution, blank)
    negative = torch.where(attribution < 0, attribution, blank)
    # Transformation: Linear transformation performs the best
    balance_attribution = positive + gamma * negative
    loss = torch.sum(balance_attribution) / \
        (base_feature.shape[0]*base_feature.shape[1])
    return loss


def normalize(grad, opt=2):
    if opt == 0:
        nor_grad = grad
    elif opt == 1:
        abs_sum = torch.sum(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        nor_grad = grad/abs_sum
    elif opt == 2:
        square = torch.sum(torch.square(grad), dim=(1, 2, 3), keepdim=True)
        nor_grad = grad/torch.sqrt(square)
    return nor_grad


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / \
        (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, kern_size):
    x = torch.nn.functional.pad(
        x, (kern_size, kern_size, kern_size, kern_size), "constant", 0)
    x = torch.nn.functional.conv2d(
        x, stack_kern, stride=1, padding=0, groups=3)
    return x


"""Input diversity: https://arxiv.org/abs/1803.06978"""


def input_diversity(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize,
                        size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(
        x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(),
                            size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(),
                             size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(
    ), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret


P_kern, kern_size = project_kern(3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_kernel = gkern(7, 3)


class FGSMGradSSA:
    def __init__(self, epsilon, data_min, data_max,N=20):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.N = N

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False, verbose=False):
        num_channels = data.shape[1]
        if num_channels != 3:
            T_kernel = gkern(7, 3, num_channels=num_channels)
        else:
            T_kernel = gkern(7, 3)
        image_width = data.shape[2]
        momentum = 1.0
        alpha = self.epsilon / num_steps
        grad = 0
        rho = 0.5
        # N = 20
        N = self.N
        sigma = 16
        # import pdb; pdb.set_trace()
        dt = data.clone().detach().requires_grad_(True)
        # print('dt first', dt.shape)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        if verbose:
            pbar = tqdm(range(num_steps))
        else:
            pbar = range(num_steps)
        # print('first loop')
        for _ in pbar:
        # for _ in tqdm(range(num_steps)):
            model.zero_grad()
            noise = 0
            if verbose:
                pbar_n = tqdm(range(N))
            else:
                pbar_n = range(N)
            # print('_', _)
            for n in pbar_n:
            # for n in tqdm(range(N)):
                gauss = torch.randn(dt.size()[0], num_channels, image_width, image_width) * (sigma / 255)
                # print('gauss', gauss.shape)
                gauss = gauss.cuda()
                # print('dt', dt.shape)
                dt_dct = dct_2d(dt + gauss).cuda()
                # print('dt_dct', dt_dct.shape)
                mask = (torch.rand_like(dt) * 2 * rho + 1 - rho).cuda()
                # print('mask', mask.shape)
                dt_idct = idct_2d(dt_dct * mask)
                # print('dt_idct', dt_idct.shape)
                dt_idct = V(dt_idct, requires_grad = True)

                # DI-FGSM https://arxiv.org/abs/1803.06978
                output_v3 = model(DI(dt_idct))
                if use_softmax:
                    output_v3 = F.softmax(output_v3, dim=-1)
                loss = F.cross_entropy(output_v3, target)
                loss.backward()
                noise += dt_idct.grad.data
                # print('dt_idct.grad.data', dt_idct.grad.data.shape)
            noise = noise / N
            # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
            # print('noise bf', noise.shape)
            # print('T_kernel', T_kernel.shape)
            # print('num_channels', num_channels)
            noise_copy = noise.clone()
            noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=num_channels)
            # print('noise af', noise.shape)
            # import pdb; pdb.set_trace()
            # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
            noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            noise = momentum * grad + noise
            grad = noise
            grad = grad.detach()
            # print('hello!! dt', dt.shape)
            for i, idx in enumerate(leave_index):
                grads[idx].append(grad[i:i+1].clone())
            if use_sign:
                # print('use_sign')
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                # print('dt', dt.shape)
                # print('data', data.shape)
                # print('total_grad', total_grad.shape)
                # print('adv_data', adv_data.shape)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                # print('not use_sign')
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                # print('dt', dt.shape)
                # print('data', data.shape)
                # print('total_grad', total_grad.shape)
                # print('adv_data', adv_data.shape)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            # print('hi!! dt', dt.shape)
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
                    grad = grad[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        adv_pred = model(dt)
        model.zero_grad()
        grad = 0
        noise = 0
        # print('second loop')
        for n in range(N):
            gauss = torch.randn(dt.size()[0], num_channels, image_width, image_width) * (sigma / 255)
            # print('gauss', gauss.shape)
            gauss = gauss.cuda()
            dt_dct = dct_2d(dt + gauss).cuda()
            # print('dt_dct', dt_dct.shape)
            mask = (torch.rand_like(dt) * 2 * rho + 1 - rho).cuda()
            # print('mask', mask.shape)
            dt_idct = idct_2d(dt_dct * mask)
            # print('dt_idct', dt_idct.shape)
            dt_idct = V(dt_idct, requires_grad = True)
            # print('dt_idct', dt_idct.shape)

            # DI-FGSM https://arxiv.org/abs/1803.06978
            output_v3 = model(DI(dt_idct))
            if use_softmax:
                output_v3 = F.softmax(output_v3, dim=-1)
            loss = F.cross_entropy(output_v3, target_clone)
            loss.backward()
            noise += dt_idct.grad.data
        noise = noise / N
        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=num_channels)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        grad = grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads


class AMPE:
    def __init__(self, model):
        self.model = model

    def __call__(self, hats, grads):
        t_list = hats[1:] - hats[:-1]
        grads = grads[:-1]
        total_grads = -torch.sum(t_list * grads, dim=0)
        attribution_map = total_grads.unsqueeze(0)
        return attribution_map.detach().cpu().numpy()
