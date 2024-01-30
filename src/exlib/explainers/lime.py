import torch
import torch.nn.functional as F
import lime
import math
import numpy as np
from lime import lime_image, lime_text, lime_base
from .common import *
# from .libs.lime.lime_text import LimeTextExplainer


def lime_cls_closure(model, collapse):
    def go(x_np):
        x = np_to_torch_img(x_np)
        x = x.to(next(model.parameters()).device)
        if collapse:
            x = x[:,0:1,:,:] # even though lime needs no singleton last dimension in its input,
            # for an odd reason they put back 3 of them to match RGB format before passing
            # to batch_predict. So we need to remove the extraneous ones.
        y = model(x)
        return y.detach().cpu().numpy()
    return go


def explain_image_cls_with_lime(model, x, ts,
                                LimeImageExplainerKwargs={},
                                # Gets FA for every label if top_labels == None
                                explain_instance_kwargs={},
                                get_image_and_mask_kwargs={}):
    """
    Explain a pytorch model with LIME.
    this function is not intended to be called directly.
    We only explain one image at a time.

    # LimeImageExplainer args
    kernel_width=0.25, kernel=None, verbose=False, feature_selection='auto', random_state=None

    # explain_instance args
    image, classifier_fn, labels=(1, ), hide_color=None, top_labels=5,
    num_features=100000, num_samples=1000, batch_size=10, segmentation_fn=None,
    distance_metric='cosine', model_regressor=None, random_seed=None, progress_bar=true

    # get_image_and_mask arguments
    positive_only=true, negative_only=False, hide_rest=False, num_features=5, min_weight=0.0
    """

    ## Images here are not batched
    C, H, W = x.shape
    x_np = x.cpu().permute(1,2,0).numpy()

    collapse = x.size(0) == 1
    if collapse:
        x_np = x_np[:,:,0]

    f = lime_cls_closure(model, collapse)
    explainer = lime_image.LimeImageExplainer(**LimeImageExplainerKwargs)

    if isinstance(ts, torch.Tensor):
        todo_labels = ts.numpy()
    else:
        todo_labels = ts

    lime_exp = explainer.explain_instance(x_np, f, labels=todo_labels, **explain_instance_kwargs)

    attrs = torch.zeros_like(x).to(x.device)
    for i, ti in enumerate(todo_labels):
        seg_mask = torch.from_numpy(lime_exp.segments).to(x.device)
        seg_attrs = lime_exp.local_exp[ti]
        for seg_id, seg_attr in seg_attrs:
            attrs += (seg_mask == seg_id) * seg_attr

    return FeatureAttrOutput(attrs, lime_exp)


class LimeImageCls(FeatureAttrMethod):
    def __init__(self, model,
                 LimeImageExplainerKwargs={},
                 explain_instance_kwargs={
                     # Make this however big you need to get every label
                     # this is because the original LIME API is stupid
                     "top_labels" : 1000000,
                     "num_samples" : 500,
                 },
                 get_image_and_mask_kwargs={}):
        super(LimeImageCls, self).__init__(model)
        self.LimeImageExplainerKwargs = LimeImageExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs
        self.get_image_and_mask_kwargs = get_image_and_mask_kwargs

    def forward(self, x, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        N = x.size(0)
        assert x.ndim == 4 and t.ndim == 1 and len(t) == N

        attrs, lime_exps = [], []
        for i in range(N):
            xi, ti = x[i], t[i].cpu().item()
            out = explain_image_cls_with_lime(self.model, xi, [ti],
                    LimeImageExplainerKwargs=self.LimeImageExplainerKwargs,
                    explain_instance_kwargs=self.explain_instance_kwargs,
                    get_image_and_mask_kwargs=self.get_image_and_mask_kwargs)

            attrs.append(out.attributions)
            lime_exps.append(out.explainer_output)

        return FeatureAttrOutput(torch.stack(attrs), lime_exps)


# Segmentation model
class LimeImageSeg(FeatureAttrMethod):
    def __init__(self, model,
                 LimeImageExplainerKwargs={},
                 explain_instance_kwargs={
                     # Make this however big you need to get every label
                     # this is because the original LIME API is stupid
                     "top_labels" : 1000000,
                     "num_samples" : 500,
                 },
                 get_image_and_mask_kwargs={}):
        super(LimeImageSeg, self).__init__(model)
        self.LimeImageExplainerKwargs = LimeImageExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs
        self.get_image_and_mask_kwargs = get_image_and_mask_kwargs

        self.cls_model = Seg2ClsWrapper(model)

    def forward(self, x, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        N = x.size(0)
        assert x.ndim == 4 and t.ndim == 1 and len(t) == N

        attrs, lime_exps = [], []
        for i in range(N):
            xi, ti = x[i], t[i].cpu().item()
            out = explain_image_cls_with_lime(self.cls_model, xi, [ti],
                    LimeImageExplainerKwargs=self.LimeImageExplainerKwargs,
                    explain_instance_kwargs=self.explain_instance_kwargs,
                    get_image_and_mask_kwargs=self.get_image_and_mask_kwargs)

            attrs.append(out.attributions)
            lime_exps.append(out.explainer_output)

        return FeatureAttrOutput(torch.stack(attrs), lime_exps)


# class LimeTextExplainerLocal(lime_text.LimeTextExplainer):
#     """Explains text classifiers.
#        Currently, we are using an exponential kernel on cosine distance, and
#        restricting explanations to words that are present in documents."""
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def explain_instance(self,
#                          text_instance,
#                          classifier_fn,
#                          labels=(1,),
#                          top_labels=None,
#                          num_features=10,
#                          num_samples=5000,
#                          distance_metric='cosine',
#                          model_regressor=None):
#         """Generates explanations for a prediction.

#         First, we generate neighborhood data by randomly hiding features from
#         the instance (see __data_labels_distance_mapping). We then learn
#         locally weighted linear models on this neighborhood data to explain
#         each of the classes in an interpretable way (see lime_base.py).

#         Args:
#             text_instance: raw text string to be explained.
#             classifier_fn: classifier prediction probability function, which
#                 takes a list of d strings and outputs a (d, k) numpy array with
#                 prediction probabilities, where k is the number of classes.
#                 For ScikitClassifiers , this is classifier.predict_proba.
#             labels: iterable with labels to be explained.
#             top_labels: if not None, ignore labels and produce explanations for
#                 the K labels with highest prediction probabilities, where K is
#                 this parameter.
#             num_features: maximum number of features present in explanation
#             num_samples: size of the neighborhood to learn the linear model
#             distance_metric: the distance metric to use for sample weighting,
#                 defaults to cosine similarity
#             model_regressor: sklearn regressor to use in explanation. Defaults
#             to Ridge regression in LimeBase. Must have model_regressor.coef_
#             and 'sample_weight' as a parameter to model_regressor.fit()
#         Returns:
#             An Explanation object (see explanation.py) with the corresponding
#             explanations.
#         """

#         indexed_string = (lime_text.IndexedCharacters(
#             text_instance, bow=self.bow, mask_string=self.mask_string)
#                           if self.char_level else
#                           lime_text.IndexedString(text_instance, bow=self.bow,
#                                         split_expression=self.split_expression,
#                                         mask_string=self.mask_string))
#         # import pdb; pdb.set_trace()
#         domain_mapper = lime_text.TextDomainMapper(indexed_string)
#         data, yss, distances = self.__data_labels_distances(
#             indexed_string, classifier_fn, num_samples,
#             distance_metric=distance_metric)
#         if self.class_names is None:
#             self.class_names = [str(x) for x in range(yss[0].shape[0])]
#         ret_exp = lime_base.explanation.Explanation(domain_mapper=domain_mapper,
#                                           class_names=self.class_names,
#                                           random_state=self.random_state)
#         ret_exp.score, ret_exp.local_pred = {}, {}
#         ret_exp.predict_proba = yss[0]
#         if top_labels:
#             labels = np.argsort(yss[0])[-top_labels:]
#             ret_exp.top_labels = list(labels)
#             ret_exp.top_labels.reverse()
#         for label in labels:
#             (ret_exp.intercept[label],
#              ret_exp.local_exp[label],
#              ret_exp.score[label],
#              ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
#                 data, yss, distances, label, num_features,
#                 model_regressor=model_regressor,
#                 feature_selection=self.feature_selection)
#         return ret_exp
    

def lime_cls_closure_text(model, tokenizer):
    def go(x_str):
        x_raw = [tokenizer.decode(tokenizer.convert_tokens_to_ids(x_str_i.split())) for x_str_i in x_str]
        inp = tokenizer(x_raw, 
                        padding='max_length', 
                        truncation=True, 
                        max_length=512)
        inp = {k: torch.tensor(v).to(next(model.parameters()).device) 
               for k, v in inp.items()}
        y = model(**inp)
        return y.detach().cpu().numpy()
    return go

def explain_text_cls_with_lime(model, tokenizer, x, ts,
                                LimeTextExplainerKwargs={},
                                # Gets FA for every label if top_labels == None
                                explain_instance_kwargs={}):
    """
    Explain a pytorch model with LIME.
    this function is not intended to be called directly.
    We only explain one text at a time.

    # LimeImageExplainer args
    kernel_width=0.25, kernel=None, verbose=False, feature_selection='auto', random_state=None

    # explain_instance args
    image, classifier_fn, labels=(1, ), hide_color=None, top_labels=5,
    num_features=100000, num_samples=1000, batch_size=10, segmentation_fn=None,
    distance_metric='cosine', model_regressor=None, random_seed=None, progress_bar=true

    # get_image_and_mask arguments
    positive_only=true, negative_only=False, hide_rest=False, num_features=5, min_weight=0.0
    """

    ## Texts here are not batched
    # L = x.shape
    tokens = tokenizer.convert_ids_to_tokens(x)
    x_str = ' '.join(tokens)

    f = lime_cls_closure_text(model, tokenizer)
    explainer = LimeTextExplainer(**LimeTextExplainerKwargs)

    if isinstance(ts, torch.Tensor):
        todo_labels = ts.numpy()
    else:
        todo_labels = ts

    lime_exp = explainer.explain_instance(x_str, f, labels=todo_labels, **explain_instance_kwargs)
    
    explanation_dict = {w: s for w, s in lime_exp.as_list()}
    
    attrs = torch.tensor([explanation_dict[token] if token in explanation_dict else 0 for token in tokens])

    return FeatureAttrOutput(attrs, lime_exp)


class LimeTextCls(FeatureAttrMethod):
    def __init__(self, model, tokenizer,
                 LimeTextExplainerKwargs={},
                 explain_instance_kwargs={
                     # Make this however big you need to get every label
                     # this is because the original LIME API is stupid
                     "top_labels" : 1000000,
                     "num_samples" : 500,
                 }):
        super().__init__(model)
        self.tokenizer = tokenizer
        self.LimeTextExplainerKwargs = LimeTextExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs

    def forward(self, x, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        N = x.size(0)
        assert x.ndim == 2 and t.ndim == 1 and len(t) == N

        attrs, lime_exps = [], []
        for i in range(N):
            xi, ti = x[i], t[i].cpu().item()
            out = explain_text_cls_with_lime(self.model, self.tokenizer, xi, [ti],
                    LimeTextExplainerKwargs=self.LimeTextExplainerKwargs,
                    explain_instance_kwargs=self.explain_instance_kwargs)

            attrs.append(out.attributions)
            lime_exps.append(out.explainer_output)

        return FeatureAttrOutput(torch.stack(attrs), lime_exps)