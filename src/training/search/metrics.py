"""
This module provides set of classes
for evaluating embedding similarity search
algorithms.
"""
import torch
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

    def calculate_rank_precision(self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> float:
        pass

    def forward(self, predicted_ranks: torch.Tensor, true_ranks: torch.Tensor):
        precision = self.calculate_rank_precision(predicted_ranks, true_ranks)
        return precision 


class MeanAveragePrecision(nn.Module):
    """
    Implementation of the Mean Average Precision at K
    evaluation metric
    """
    def __init__(self, k: int, conf_threshold: float):
        super(MeanAveragePrecision, self).__init__()
        self.conf_threshold = conf_threshold 
        self.k = k

class MeanAverageRecall(nn.Module):
    """
    Implementation of the Mean Average Recall at K
    evaluation metric
    """
    def __init__(self, k: int, conf_threshold: float):
        super(MeanAverageRecall, self).__init__()
        self.k = k
        self.conf_threshold = conf_threshold

    def forward(self, 
        predicted_ranks: torch.Tensor, 
        true_ranks: torch.Tensor
    ):
        pass
