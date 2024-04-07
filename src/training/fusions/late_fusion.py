from torch import nn
import torch
import typing

class LateFusion(nn.Module):
    """
    Implementation of the Late Fusion
    module for handling embeddings 
    from multiple modalities of the data.
    
    Parameters:
    -----------
        fusion_module - fusion strategy (additive or multiplicative)
    """
    def __init__(self, 
        fuse_module: nn.Module, 
        classifiers: typing.List[nn.Module]
    ):
        super(LateFusion, self).__init__()
        self.fuse_module = fuse_module
        self.classifiers = classifiers

    def forward(self, modalities: list):
        embeddings = torch.zeros(size=(len(modalities)))
        for idx, modality in enumerate(modalities):
            embeddings[idx] = self.classifiers[idx](modality)
        fused = self.fuse_module(embeddings)
        return fused