from torch import nn
from pytorch_metric_learning.losses import (
    ContrastiveLoss, 
    TripletMarginLoss
)
from pytorch_toolbelt.losses import FocalLoss
import torch

class KLDivergenceLoss(nn.Module):
    """
    Implementation of the Kullback-Leiber
    Divergence Loss to measure difference
    between two distributions of image pixels
    """
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, input_dist: torch.Tensor, output_dist: torch.Tensor):
        return torch.sum(
            input_dist * torch.log2(input_dist / output_dist)
        )

class ContrastLoss(nn.Module):
    """
    Loss function for contrastive learning
    of the embedding generation networks
    """
    def __init__(self, 
        epsilon: float, 
        pos_margin: float, 
        neg_margin: float
    ):
        super(ContrastiveLoss, self).__init__()
        self.epsilon = epsilon
        self.cont_loss = ContrastiveLoss(
            pos_margin=pos_margin, 
            neg_margin=neg_margin,

        )
    
    def forward(self, pred_embs: torch.Tensor, labels: torch.Tensor):
        return self.cont_loss(
            pred_embs, 
            labels
        )
    
class TripletLoss(TripletMarginLoss):
    """
    Implementation of the Triplet Loss
    Function for training multimodal embedding encoders
    """
    def __init__(self,
        margin: float, 
        smooth_loss: bool = False, 
        triplets_per_anchor='all'
    ):
        super(TripletLoss, self).__init__(
            margin=margin,
            smooth_loss=smooth_loss,
            triplets_ber_anchor=triplets_per_anchor
        )

class MultilabelFocalLoss(FocalLoss):
    """
    Multiclass Focal Loss for better 
    convergence on unbalanced datasets.
    """
    def __init__(self, 
        alpha = None,
        gamma: float = 2,
        normalized=False,
        ignore_index: bool = False
    ):
        super(FocalLoss, self).__init__(
            alpha=alpha,
            gamma=gamma,
            normalized=normalized,
            ignore_index=ignore_index
        )

class NTXentLoss(nn.Module):
    """
    Implementation of the NT Xent loss
    function for contrastive learning
    applications.
    """
    def __init__(self, 
        similarity_metric: typing.Callable, 
        temperature: float
    ):
        super(NTXentLoss, self).__init__()
        self.sim = similarity_metric
        self.temperature: float = temperature

    def forward(self, pairs: typing.List):
        sample_losses = []
        for curr_idx, (pos_sample, sample, neg_sample) in enumerate(pairs):
            rest_negatives = [
                torch.exp(self.sim(neg, sample) / self.temperature)
                for idx, (_, _, neg) in enumerate(pairs) if idx != curr_idx
            ]
            sample_loss = -torch.log10(
                torch.exp(self.sim(pos_sample, sample) / self.temperature
                ) / torch.sum(rest_negatives)
            )
            sample_losses.append(sample_loss)
        return torch.mean(sample_losses)

