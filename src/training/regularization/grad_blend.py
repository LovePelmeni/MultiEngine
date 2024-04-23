from src.training.trainers import base as base_trainer
import logging

logger = logging.getLogger(__name__)

class GradientBlending(object):
    """
    Implementation of the Gradient Blending
    for stabilized and robust training of multimodal
    networks.

    Paper: https://arxiv.org/pdf/1905.12681.pdf
    """
    def __init__(self):
        self.prev_validation_losses: torch.Tensor = None
        self.prev_train_losses: torch.Tensor = None

    def compute_weights(self,
        curr_valid_losses: torch.Tensor, 
        curr_train_losses: torch.Tensor) -> torch.Tensor:

        if self.prev_validation_losses is None or self.prev_train_losses is None:
            self.prev_validation_losses = curr_valid_losses
            self.curr_train_losses = curr_train_losses 
            return torch.ones_like(curr_valid_losses)
        else:
            On = self.prev_validation_losses - self.prev_train_losses
            ONn = curr_valid_losses - curr_train_losses
            dO = ONn - On
            dG = curr_valid_losses - self.prev_validation_losses
            return (ONn - on) / dG




