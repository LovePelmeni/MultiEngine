from pytorch_metric_learning.losses import (
    TripletMarginLoss,
    ContrastiveLoss,
)
from torch import nn
import torch

class ContrastLoss(nn.Module):
    """
    Implementation of the Contrastive
    Loss function for "Contrastive Learning" tasks.
    """
    def __init__(self, pos_margin: float = 0, neg_margin: float = 1):
        super(ContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin 
        self.neg_margin = neg_margin
    
    def forward(self):
        return ContrastiveLoss(
            pos_margin=self.pos_margin,
            neg_margin=self.neg_margin
        )

class TripletLoss(nn.Module):
    """
    Implementation of the Triplet Loss function
    for "Contrastive Learning" tasks.
    """
    def __init__(self, 
        margin: float, 
        triplets_per_anchor: int,
        swap: bool = False, 
        smooth_loss: bool = False
    ):
        super(TripletMarginLoss, self).__init__()
        self.loss = TripletMarginLoss(
            margin=margin, 
            swap=swap, 
            smooth_loss=smooth_loss,
            triplets_per_anchor=triplets_per_anchor,
        )
    
    def forward(self, pred_emb: torch.Tensor, true_emb: torch.Tensor, hard_pairs: tuple):
        return self.loss(
            pred_emb, 
            true_emb, 
            hard_pairs
        )