from torch import nn

class KLDivergenceLoss(nn.Module):
    """
    Implementation of the Kullback-Leiber
    Divergence Loss to measure difference
    between two distributions of image pixels
    """
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

class ContrastiveLoss(nn.Module):
    """
    Loss function for contrastive learning
    of the embedding generation networks
    """
    def __init__(self, epsilon: float):
        super(ContrastiveLoss, self).__init__()
        self.epsilon = epsilon
    
class TripletLoss(nn.Module):
    """
    Implementation of the Triplet Loss
    """
    def __init__(self, alpha: float):
        super(TripletLoss, self).__init__()
        self.alpha = alpha