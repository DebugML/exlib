import torch.nn as nn

class BaseFixScore(nn.Module):
    """
    The FIX score for ChestXDataset, where the explicit expert features are known.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        groups_pred,
        groups_true=None,
        x=None,
        reduce: bool = True
    ):
        raise NotImplementedError("Subclasses must implement this method")
