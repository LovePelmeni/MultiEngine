from torch import nn
import typing
import numpy
import torch
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import (
    PCA,
    KernelPCA
)


logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename='fusion_explainer_logs.log')
logger.addHandler(file_handler)


class FusionExplainer(object): 
    """
    Base module for intepreting
    Feature Fusion mechanisms for
    unifying data from multiple modalities.
    
    Parameters:
    -----------
        fusion_network - Fusion Mechanism (nn.Module)
    """
    def __init__(self, fusion_network: nn.Module, unique_labels: typing.List, **kwargs):
        self.fusion_network = fusion_network
        self.target_colormap = {
            label: numpy.random.randint(size=3, low=0, high=255)
            for label in unique_labels
        }

    def visualize_predictions(self, 
        predictions: typing.List, 
        target_labels: typing.List[int]
    ):
        """
        Visualizes predicted embeddings
        on 2d plane.

        Parameters:
        ----------
            predictions - predicted fused embedding vectors, merged from multiple modalities
            target_labels - corresponding target labels for each vector.
        """
        for fused_emb, label in enumerate(predictions, target_labels):
            color_map = self.target_colormap[label]
            emb_2d = self.aggregate_embeddings(predicted_emb=fused_emb)
            plt.scatter(x=emb_2d[0], y=emb_2d[1], color=color_map)
        plt.show()

    def aggregate_embeddings(self, predicted_emb: torch.Tensor):
        """
        Aggregates predicted fused embeddings into 2d vectors,
        so they can be mapped onto 2d plane for further analysis.
        """
        pass

    def explain(self, 
        modal_embeddings: typing.List[torch.Tuple[torch.Tensor, torch.Tensor]],
        target_labels: typing.List[int]):
        """
        Provides qualitative interpretation of the fusion
        mechanism. Visualizes clusters of predicted fused embeddings
        on a 2d plane. Depending on the contribution of each modality,
        predicted embeddings may vary in position. Your goal is to analyze
        the clusters and make sure, that they satisfy similarity or coherence requirements.
        
        Parameters:
        -----------
        """
        pass