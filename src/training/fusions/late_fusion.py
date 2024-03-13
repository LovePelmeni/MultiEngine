from torch import nn
import torch
import typing

class LateFusion(nn.Module):
    """
    Implementation of the Late Fusion
    module for handling embeddings 
    from multiple modalities of the data.
    """
    def __init__(self):
        super(LateFusion, self).__init__()

    def forward(self, 
        fuse_module: nn.Module, 
        classifiers: typing.List[nn.Module],
        modalities: list
    ):
        embeddings = torch.zeros(size=(len(modalities)))
        for idx, modality in enumerate(modalities):
            embeddings[idx] = classifiers[idx](modality)
        fused = fuse_module(embeddings)
        return fused