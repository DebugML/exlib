from .core import pgd_step,DL, BIG, FGSM, MFABA, MFABACOS, MFABANORM, FGSMGradSingle, FGSMGrad, IntegratedGradient, SaliencyGradient, SmoothGradient, AMPE, FGSMGradSSA
import torch
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def agi(model, data, target, epsilon=0.05, max_iter=20, topk=1):
    model = model[:2]
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = random.sample(list(range(0, 999)), topk)
    output = model(data)

    init_pred = output.argmax(-1)

    top_ids = selected_ids

    step_grad = 0

    for l in top_ids:

        targeted = torch.tensor([l] * data.shape[0]).to(device)

        if l < 999:
            targeted[targeted == init_pred] = l + 1
        else:
            targeted[targeted == init_pred] = l - 1

        delta, perturbed_image = pgd_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy()
    return adv_ex


def big(model, data, target, data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map


def mfaba_smooth(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    attack = FGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map


def mfaba_sharp(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255, use_sign=False, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    attack = FGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map


def mfaba_cos(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255, use_sign=False, use_softmax=False):
    mfaba_cos = MFABACOS(model)
    attack = FGSMGradSingle(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = mfaba_cos(data, dt, hats, grads)
    return attribution_map


def mfaba_norm(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255, use_sign=False, use_softmax=False):
    mfaba_norm = MFABANORM(model)
    attack = FGSMGradSingle(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, success, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = mfaba_norm(data, dt, hats, grads)
    return attribution_map


def ig(model, data, target, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ig = IntegratedGradient(model)
    return ig(data, target, gradient_steps=gradient_steps)


def sm(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sm = SaliencyGradient(model)
    return sm(data, target)


def sg(model, data, target, stdevs=0.15, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sg = SmoothGradient(model, stdevs=stdevs)
    return sg(data, target, gradient_steps=gradient_steps)


def deeplift(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    dl = DL(model)
    return dl(data, target)

def ampe(model, data, target, data_min=0, data_max=1, epsilon=16,N=20,num_steps=10, use_sign=True, use_softmax=True, verbose=False):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = AMPE(model)
    epsilon = epsilon / 255
    attack = FGSMGradSSA(
        epsilon=epsilon, data_min=data_min, data_max=data_max,N=N)
    _, _, _, hats, grads = attack(
        model, data, target,num_steps=num_steps, use_sign=use_sign, use_softmax=use_softmax, verbose=verbose)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map