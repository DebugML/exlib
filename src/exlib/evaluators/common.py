import torch
import torch.nn as nn


class Evaluator(nn.Module): 
    """ Explaination methods that create feature attributions should follow 
    this signature. """
    def __init__(self, model, postprocess=None): 
        super(Evaluator, self).__init__() 
        self.model = model
        self.postprocess = postprocess

    def forward(self, X, Z): 
        """ Given a minibatch of examples X and feature attributions Z, 
        evaluate the quality of the feature attribution. """
        raise NotImplementedError()


def convert_idx_masks_to_bool(masks):
    """
    input: masks (1, img_dim1, img_dim2) or (1, seq_len)
    output: masks_bool (num_masks, img_dim1, img_dim2) or (num_masks, seq_len)
    """
    unique_idxs = torch.sort(torch.unique(masks)).values
    if len(masks.shape) == 3:
        idxs = unique_idxs.view(-1, 1, 1)
        broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                        masks.shape[1], 
                                        masks.shape[2])
    else:
        idxs = unique_idxs.view(-1, 1)
        broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                        masks.shape[1])
    masks_bool = (broadcasted_masks == idxs)
	
    return masks_bool


# text
def apply_mask(kwargs, mask, mode='mult'):
    kwargs_masked = {}
    assert mode in ['mult']
    if mode == 'mult':
        for k, v in kwargs.items():
            kwargs_masked[k] = v * mask.to(v.dtype)
    else:
        raise ValueError(f'Mode {mode} not in the list of modes.')
        # inputs_new = inputs * mask.long()
    return kwargs_masked

def masked_predict(inputs, mask, model, mode='mult', kwargs={}):
    inputs_masked = apply_mask({'inputs': inputs}, mask, mode)['inputs']
    kwargs_masked = apply_mask(kwargs, mask, mode)
    logits_masked = model(inputs_masked, **kwargs_masked)
    return logits_masked

def predict(logits):
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs)
    return probs, pred

def kl_div(probs_output, probs_target):
    return torch.nn.functional.kl_div(torch.log(probs_output), probs_target, reduction='sum')
