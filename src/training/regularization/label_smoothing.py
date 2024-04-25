from torch import nn

class LabelSmoothing(nn.Module):
    """
    Implementation of the Label Smoothing
    regularization, tailored to reduce 
    influence noise, presented in the data on 
    training of big and confident DL networks.

    Parameters:
    -----------
        num_classes - number of classes, used
        for classification, or presented in the train dataset.

        epsilon - small constant for regulating proportion of the
        noise to add to labels, by default is 0.1
    """
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        self.num_classes = num_classes
        self.epsilong = epsilon

    def forward(self, one_hot_vector: torch.Tensor):
        return torch.where(
            one_hot_vector == 1,
            1 - self.epsilon,
            self.epsilon / (self.num_classes - 1)
        )