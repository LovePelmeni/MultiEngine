"""
This module provides set of classes
for evaluating embedding similarity search
algorithms.
"""
import torch
import typing
from torch import nn

class MeasureFactorialRank(nn.Module):
    """
    Base Implementation of the Measure Factorial Rank (MRR)
    evaluation metric.

    Parameters:
    -----------
        k - number of measures.
    """ 
    def __init__(self, k: int):
        super(MeasureFactorialRank, self).__init__()
        self.k = k 

    def forward(self, predicted_ranks: torch.Tensor):
        ranked_sum = torch.sum(1 / predicted_ranks) / len(predicted_ranks)
        return ranked_sum

class RecallAtK(nn.Module):
    """
    Base implementation of the Recall at K 
    evaluation metric.

    Parameters:
    ----------
        k - number of observations
    """
    def __init__(self, k: int):
        self.k_measures = k

    def calculate_rank_recall(self, 
        predicted_ranks: torch.Tensor, 
        true_ranks: torch.Tensor
    ):
        pass 

    def forward(self, 
        predicted_ranks: torch.Tensor, 
        true_ranks: torch.Tensor
    ):
        recall = self.calculate_rank_recall(predicted_ranks, true_ranks)
        return recall

class PrecisionAtK(nn.Module):
    """
    Base Implementation of the Precision at K 
    evaluation metric.

    Parameters:
    ----------
        k - number of observations
    """
    def __init__(self, k: int):
        super(PrecisionAtK, self).__init__()
        self.k_measures = k

    def calculate_precision(self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> float:
        pass

    def forward(self, predicted_ranks: torch.Tensor, true_ranks: torch.Tensor):
        precision = self.calculate_rank_precision(predicted_ranks, true_ranks)
        return precision


class TopK(nn.Module):
    """
    Implementation of the top K similarity
    evaluation metric.

    Parameters:
    -----------
        n_neighbors (int) - number of nearest neighbors to search for in the dataset.
    """
    def __init__(self, n_neighbors: int):
        super(TopK, self).__init__()
        self.n_neighbors = n_neighbors 

    def forward(self,
        sorted_pred_labels: typing.List[int], 
        target_label: int
    ):
        top_k_labels = torch.as_tensor(sorted_pred_labels[:self.n_neighbors])
        top_k_labels.requires_grad = False
        accuracy = (
            top_k_labels[top_k_labels == target_label].to(
            torch.uint8).sum() / self.n_neighbors
        )
        return accuracy