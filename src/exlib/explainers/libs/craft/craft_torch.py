
"""
CRAFT Module for Tensorflow
"""

from abc import ABC, abstractmethod
from math import ceil
from typing import Callable, Optional

import torch
import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import NotFittedError

# Anton: Comment these two out to eliminate dependency on T E N S O R F L O W
# from .sobol.sampler import HaltonSequence
# from .sobol.estimators import JansenEstimator


""" Anton: Copied from sobol.sampler.py """

class Sampler(ABC):
    """
    Base class for replicated design sampling.
    """

    @staticmethod
    def build_replicated_design(sampling_a, sampling_b):
        """
        Build the replicated design matrix C using A & B

        Parameters
        ----------
        sampling_a
          The masks values for the sampling matrix A.
        sampling_b
          The masks values for the sampling matrix B.

        Returns
        -------
        replication_c
          The new replicated design matrix C generated from A & B.
        """
        replication_c = np.array([sampling_a.copy() for _ in range(sampling_a.shape[-1])])
        for i in range(len(replication_c)):
            replication_c[i, :, i] = sampling_b[:, i]

        replication_c = replication_c.reshape((-1, sampling_a.shape[-1]))

        return replication_c

    @abstractmethod
    def __call__(self, dimension, nb_design):
        raise NotImplementedError()


class ScipySampler(Sampler):
    """
    Base class based on Scipy qmc module for replicated design sampling.
    """

    def __init__(self):
        try:
            self.qmc = scipy.stats.qmc # pylint: disable=E1101
        except AttributeError as err:
            raise ModuleNotFoundError("Xplique need scipy>=1.7 to use this sampling.") from err


""" Anton: Copied from sobol.estimators.py """

class SobolEstimator(ABC):
    """
    Base class for Sobol' total order estimators.
    """

    @staticmethod
    def masks_dim(masks):
        """
        Deduce the number of dimensions using the sampling masks.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        nb_dim
          The number of dimensions under study according to the masks.
        """
        nb_dim = np.prod(masks.shape[1:])
        return nb_dim

    @staticmethod
    def split_abc(outputs, nb_design, nb_dim):
        """
        Split the outputs values into the 3 sampling matrices A, B and C.

        Parameters
        ----------
        outputs
          Model outputs for each sample point of matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).
        nb_dim
          Number of dimensions to estimate.

        Returns
        -------
        a
          The results for the sample points in matrix A.
        b
          The results for the sample points in matrix A.
        c
          The results for the sample points in matrix C.
        """
        sampling_a = outputs[:nb_design]
        sampling_b = outputs[nb_design:nb_design*2]
        replication_c = np.array([outputs[nb_design*2 + nb_design*i:nb_design*2 + nb_design*(i+1)]
                      for i in range(nb_dim)])
        return sampling_a, sampling_b, replication_c

    @staticmethod
    def post_process(stis, masks):
        """
        Post processing ops on the indices before sending them back. Makes sure the data
        format and shape is correct.

        Parameters
        ----------
        stis
          Total order Sobol' indices, one for each dimensions.
        masks
            Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        stis
          Total order Sobol' indices after post processing.
        """
        stis = np.array(stis, np.float32)
        return stis.reshape(masks.shape[1:])

    @abstractmethod
    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Ref. Jansen, M., Analysis of variance designs for model output (1999)
        https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        raise NotImplementedError()


class JansenEstimator(SobolEstimator):
    """
    Jansen estimator for total order Sobol' indices.

    Ref. Jansen, M., Analysis of variance designs for model output (1999)
    https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544
    """

    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self.masks_dim(masks)
        sampling_a, _, replication_c = self.split_abc(outputs, nb_design, nb_dim)

        mu_a = np.mean(sampling_a)
        var = np.sum([(v - mu_a)**2 for v in sampling_a]) / (len(sampling_a) - 1)

        stis = [
            np.sum((sampling_a - replication_c[i])**2.0) / (2 * nb_design * var)
            for i in range(nb_dim)
        ]

        return self.post_process(stis, masks)


""" Anton: Original contents of craft_torch.py """


def torch_to_numpy(tensor):
  try:
    return tensor.detach().cpu().numpy()
  except:
    return np.array(tensor)


def _batch_inference(model, dataset, batch_size=128, resize=None, device='cuda'):
  nb_batchs = ceil(len(dataset) / batch_size)
  start_ids = [i*batch_size for i in range(nb_batchs)]

  results = []

  with torch.no_grad():
    for i in start_ids:
      x = torch.tensor(dataset[i:i+batch_size])
      x = x.to(device)

      if resize:
        x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)

      results.append(model(x).cpu())

  results = torch.cat(results)
  return results


class BaseConceptExtractor(ABC):
    """
    Base class for concept extraction models.

    Parameters
    ----------
    input_to_latent : Callable
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
    latent_to_logit : Callable
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    number_of_concepts : int
        The number of concepts to extract.
    batch_size : int, optional
        The batch size to use during training and prediction. Default is 64.

    """

    def __init__(self, input_to_latent : Callable,
                       latent_to_logit : Optional[Callable] = None,
                       number_of_concepts: int = 20,
                       batch_size: int = 64):

        # sanity checks
        assert(number_of_concepts > 0), "number_of_concepts must be greater than 0"
        assert(batch_size > 0), "batch_size must be greater than 0"
        assert(callable(input_to_latent)), "input_to_latent must be a callable function"

        self.input_to_latent = input_to_latent
        self.latent_to_logit = latent_to_logit
        self.number_of_concepts = number_of_concepts
        self.batch_size = batch_size

    @abstractmethod
    def fit(self, inputs):
        """
        Fit the CAVs to the input data.

        Parameters
        ----------
        inputs : array-like
            The input data to fit the model on.

        Returns
        -------
        tuple
            A tuple containing the input data and the matrices (U, W) that factorize the data.

        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, inputs):
        """
        Transform the input data into a concepts embedding.

        Parameters
        ----------
        inputs : array-like
            The input data to transform.

        Returns
        -------
        array-like
            The transformed embedding of the input data.

        """
        raise NotImplementedError


class Craft(BaseConceptExtractor):
    """
    Class Implementing the CRAFT Concept Extraction Mechanism.

    Parameters
    ----------
    input_to_latent : Callable
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
    latent_to_logit : Callable, optional
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    number_of_concepts : int
        The number of concepts to extract.
    batch_size : int, optional
        The batch size to use during training and prediction. Default is 64.
    patch_size : int, optional
        The size of the patches to extract from the input data. Default is 64.
    """

    def __init__(self, input_to_latent: Callable,
                       latent_to_logit: Optional[Callable] = None,
                       number_of_concepts: int = 20,
                       batch_size: int = 64,
                       patch_size: int = 64,
                       device : str = 'cuda'):
        super().__init__(input_to_latent, latent_to_logit, number_of_concepts, batch_size)

        self.patch_size = patch_size
        self.activation_shape = None
        self.device = device


    def fit(self, inputs: torch.Tensor, return_patch_masks: bool = False):
        """
        Fit the Craft model to the input data.

        Parameters
        ----------
        inputs : np.ndarray
            Preprocessed Iinput data of shape (n_samples, channels, height, width).
            (x1, x2, ..., xn) in the paper.

        Returns
        -------
        (X, U, W)
            A tuple containing the crops (X in the paper),
            the concepts values (U) and the concepts basis (W).
        """


        assert len(inputs.shape) == 4, "Input data must be of shape (n_samples, channels, height, width)."
        assert inputs.shape[2] == inputs.shape[3], "Input data must be square."

        image_size = inputs.shape[2]

        # extract patches from the input data, keep patches on cpu
        strides = int(self.patch_size * 0.80)

        patches = torch.nn.functional.unfold(inputs, kernel_size=self.patch_size, stride=strides)
        patches = patches.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size)

        # Anton: we want to extract the masks for each patch
        N, _, H, W = inputs.shape
        K = self.patch_size
        masks = torch.zeros(H//strides, W//strides, H, W)
        for h in range(H//strides):
            for w in range(W//strides):
                masks[h, w, h*strides:h*strides+K, w*strides:w*strides+K] = torch.ones((K, K))
        masks = masks.view(-1,H,W)

        # encode the patches and obtain the activations
        activations = _batch_inference(self.input_to_latent, patches, self.batch_size, image_size, 
                                       device=self.device)

        assert torch.min(activations) >= 0.0, "Activations must be positive."

        # if the activations have shape (n_samples, height, width, n_channels),
        # apply average pooling
        if len(activations.shape) == 4:
            activations = torch.mean(activations, dim=(2, 3))

        # apply NMF to the activations to obtain matrices U and W
        reducer = NMF(n_components=self.number_of_concepts)
        U = reducer.fit_transform(torch_to_numpy(activations))
        W = reducer.components_.astype(np.float32)

        # store the factorizer and W as attributes of the Craft instance
        self.reducer = reducer
        self.W = np.array(W, dtype=np.float32)

        # Anton: adjust shapes
        patches = patches.view(N,-1,3,K,K).to(inputs.device)
        U = torch.tensor(U).view(N,-1,self.number_of_concepts).to(inputs.device)
        W = torch.tensor(W).to(inputs.device)
        masks = masks.to(inputs.device)

        if return_patch_masks:
            return patches, U, W, masks
        else:
            return patches, U, W

    def check_if_fitted(self):
        """Checks if the factorization model has been fitted to input data.

        Raises
        ------
        NotFittedError
            If the factorization model has not been fitted to input data.
        """

        if not hasattr(self, 'reducer'):
            raise NotFittedError("The factorization model has not been fitted to input data yet.")

    def transform(self, inputs: np.ndarray, activations: Optional[np.ndarray] = None):
        self.check_if_fitted()

        if activations is None:
            activations = _batch_inference(self.input_to_latent, inputs, self.batch_size,
                                           device=self.device)

        is_4d = len(activations.shape) == 4

        if is_4d:
            # (N, C, W, H) -> (N * W * H, C)
            activation_size = activations.shape[-1]
            activations = activations.permute(0, 2, 3, 1)
            activations = torch.reshape(activations, (-1, activations.shape[-1]))

        W_dtype = self.reducer.components_.dtype
        U = self.reducer.transform(torch_to_numpy(activations).astype(W_dtype))

        if is_4d:
          # (N * W * H, R) -> (N, W, H, R)
          U = np.reshape(U, (-1, activation_size, activation_size, U.shape[-1]))

        return U

    def estimate_importance(self, inputs, class_id, nb_design=32):
        """
        Estimates the importance of each concept for a given class.

        Parameters
        ----------
        inputs : numpy array or Tensor
            The input data to be transformed.
        class_id : int
            The class id to estimate the importance for.
        nb_design : int, optional
            The number of design to use for the importance estimation. Default is 32.

        Returns
        -------
        importances : list
            The Sobol total index (importance score) for each concept.

        """
        self.check_if_fitted()

        U = self.transform(inputs)

        masks = HaltonSequence()(self.number_of_concepts, nb_design=nb_design).astype(np.float32)
        estimator = JansenEstimator()

        importances = []

        if len(U.shape) == 2:
            # apply the original method of the paper

            for u in U:
                u_perturbated = u[None, :] * masks
                a_perturbated = u_perturbated @ self.W

                y_pred = _batch_inference(self.latent_to_logit, a_perturbated, self.batch_size,
                                          device=self.device)
                y_pred = y_pred[:, class_id]

                stis = estimator(torch_to_numpy(masks),
                                 torch_to_numpy(y_pred),
                                 nb_design)

                importances.append(stis)

        elif len(U.shape) == 4:
            # apply a re-parameterization trick and use mask on all localization for a given
            # concept id to estimate sobol indices
            for u in U:
                u_perturbated = u[None, :] * masks[:, None, None, :]
                a_perturbated = np.reshape(u_perturbated,(-1, u.shape[-1])) @ self.W
                a_perturbated = np.reshape(a_perturbated, (len(masks), U.shape[1], U.shape[2], -1))
                a_perturbated = np.moveaxis(a_perturbated, -1, 1)

                y_pred = _batch_inference(self.latent_to_logit, a_perturbated, self.batch_size,
                                          device=self.device)
                y_pred = y_pred[:, class_id]

                stis = estimator(torch_to_numpy(masks),
                                 torch_to_numpy(y_pred),
                                 nb_design)

                importances.append(stis)

        return np.mean(importances, 0)
