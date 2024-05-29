import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .common import *
from .libs.archipelago.explainer import Archipelago
from .libs.archipelago.application_utils.image_utils import *
from .libs.archipelago.application_utils.utils_torch import ModelWrapperTorch
from .common import patch_segmenter
import warnings
warnings.filterwarnings("ignore")


class ArchipelagoImageCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model, top_k=5, segmenter='quickshift'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_wrapper = ModelWrapperTorch(model, device)
        super().__init__(model_wrapper)
        self.top_k = top_k
        self.segmenter = segmenter

    def forward(self, x, t, **kwargs):
        bsz = x.shape[0]

        if not isinstance(t, torch.Tensor) and t is not None:
            t = torch.tensor(t)

        expln_scores_all = []
        expln_flat_masks_all = []
        masks_all = []
        mask_weights_all = []
        for i in range(bsz):
            image = x[i].cpu().permute(1,2,0).numpy()
            ti = t[i] if t is not None else None
            if ti is None:
                predictions = self.model(np.expand_dims(image,0))
                class_idx = predictions[0].argsort()[::-1][0]
            else:
                if len(t[i].shape) == 0 or len(t[i].shape) == 1 and t[i].shape[0] == 1:
                    class_idx = [t[i].cpu().item()]
                else:
                    class_idx = t[i].cpu().numpy().tolist()

            baseline = np.zeros_like(image)
            if self.segmenter == 'quickshift':
                segments = quickshift(image, kernel_size=3, max_dist=300, ratio=0.2)
            elif self.segmenter == 'patch':
                segments = patch_segmenter(image, sz=(8,8))
            else:
                segments = self.segmenter(image)

            xf = ImageXformer(image, baseline, segments)

            apgo = Archipelago(self.model, data_xformer=xf, output_indices=class_idx, batch_size=20)
            explanation = apgo.explain(top_k=self.top_k)

            expln_scores_i = []
            expln_flat_masks_i = []
            masks_i = []
            mask_weights_i = []
            for c_i in range(len(class_idx)):
                expln_scores = np.zeros_like(segments, dtype=float)
                expln_flat_masks = np.zeros_like(segments, dtype=float)
                masks = []
                mask_weights = []

                for e_i, (k, v) in enumerate(sorted(explanation[c_i].items(), key=lambda item: item[1], reverse=True)):
                    mask = np.zeros_like(segments, dtype=float)
                    for s_i in k:
                        expln_scores[segments == s_i] = v
                        expln_flat_masks[segments == s_i] = e_i
                        mask[segments == s_i] = 1
                    masks.append(mask)
                    mask_weights.append(v)

                expln_scores = torch.tensor(expln_scores).unsqueeze(0).to(x.device)
                expln_flat_masks = torch.tensor(expln_flat_masks).unsqueeze(0).to(x.device)
                masks = torch.tensor(masks).to(x.device)
                mask_weights = torch.tensor(mask_weights).to(x.device)

                expln_scores_i.append(expln_scores)
                expln_flat_masks_i.append(expln_flat_masks)
                masks_i.append(masks)
                mask_weights_i.append(mask_weights)

            expln_scores_all.append(torch.cat(expln_scores_i, dim=0))
            expln_flat_masks_all.append(torch.cat(expln_flat_masks_i, dim=0))
            masks_all.append(masks_i)
            mask_weights_all.append(mask_weights_i)

        expln_scores = torch.stack(expln_scores_all, dim=0)
        expln_flat_masks = torch.stack(expln_flat_masks_all, dim=0)

        return FeatureAttrOutput(expln_scores, {
            "expln_flat_masks": expln_flat_masks,
            "masks": masks_all,
            "mask_weights": mask_weights_all
        })


class TextXformer:
    def __init__(self, input_ids, baseline_ids):
        self.input = input_ids
        self.baseline = baseline_ids
        self.num_features = input_ids.shape[-1]
        
    def simple_xform(self, instance_repr):
        mask_indices = np.argwhere(instance_repr==True).flatten()
        id_list = list(self.baseline)
        for i in mask_indices:
            id_list[i] = self.input[i]
        return id_list

    def __call__(self, instance_repr):
        return self.simple_xform(instance_repr)
    

class BertWrapperTorch:
    def __init__(self, model, device, merge_logits=False):
        self.model = model.to(device)
        self.device = device
        self.merge_logits = merge_logits

    def get_predictions(self, batch_ids):
        batch_ids = torch.LongTensor(batch_ids).to(self.device)
        batch_conf = self.model(batch_ids)
        
        if isinstance(batch_conf, tuple):
            return batch_conf[0].data.cpu()
        else:
            if not isinstance(batch_conf, torch.Tensor):
                batch_conf = batch_conf.logits
            return batch_conf.data.cpu()
        return batch_conf

    def __call__(self, batch_ids):
        batch_predictions = self.get_predictions(batch_ids)
        if self.merge_logits:
            batch_predictions2 = (
                (batch_predictions[:, 1] - batch_predictions[:, 0]).unsqueeze(1).numpy()
            )
            return batch_predictions2
        else:
            return batch_predictions.numpy()


class ArchipelagoTextCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model, top_k=5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_wrapper = BertWrapperTorch(model, device)
        super().__init__(model_wrapper)
        self.top_k = top_k

    def forward(self, x, t, **kwargs):
        bsz = x.shape[0]

        if not isinstance(t, torch.Tensor) and t is not None:
            t = torch.tensor(t)

        masks_all = []
        mask_weights_all = []
        expln_scores = torch.zeros_like(x).float()
        for i in range(bsz):
            if 'attention_mask' in kwargs:
                text_len = kwargs['attention_mask'][i].sum()
                x_i = x[i][:text_len]
                kwargs_i = {k: v[i][:text_len] for k, v in kwargs.items()}
            else:
                x_i = x[i]
                kwargs_i = {k: v[i] for k, v in kwargs.items()}
            if t is None:
                predictions = self.model(np.expand_dims(x_i,0))
                class_idx = predictions[0].argsort()[::-1][0]
            else:
                class_idx = t[i].cpu().item()

            inputs_np = x_i.cpu().numpy()
            baseline = np.zeros_like(inputs_np)

            xf = TextXformer(inputs_np, baseline)
            apgo = Archipelago(self.model, data_xformer=xf, output_indices=class_idx, batch_size=20)
            explanation = apgo.explain(top_k=self.top_k)

            mask_weights = []
            masks = torch.zeros_like(x_i, dtype=float)

            for e_i, (k, v) in enumerate(sorted(explanation.items(), key=lambda item: item[1], reverse=True)):
                for s_i in k:
                    expln_scores[i][s_i] = float(v)
                    masks[s_i] = e_i
                mask_weights.append(v)

            mask_weights = torch.tensor(mask_weights).to(x.device)
            masks_all.append(masks)
            mask_weights_all.append(mask_weights)

        return FeatureAttrOutput(expln_scores, {
            "expln_flat_masks": masks_all,
            "masks": masks_all,
            "mask_weights": mask_weights_all,
        })