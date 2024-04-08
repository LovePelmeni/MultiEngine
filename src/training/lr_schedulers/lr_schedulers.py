from torch.optim.lr_scheduler import _LRScheduler 
from bisect import bisect_right
import typing
import numpy

class MultiStepLRScheduler(_LRScheduler):
    """
    Implementation of the Multi Step Learning Rate
    Scheduler.

    Parameters:
    -----------
        steps - steps to perform update at.
        gamma - reduction factor
    """
    def __init__(self, steps: list, gamma: float):
        self.steps = steps 
        self.gamma = gamma
        
    def get_last_lr(self) -> typing.List[float]:
        return [base_lr * numpy.power(
            self.gamma, 
            (bisect_right(self.steps, self.last_epoch)+1)
        ) for base_lr in self.base_lrs
        ]

class StepLRScheduler(_LRScheduler):
    """
    Implementation of the standard Step Learning Rate
    Scheduler.

    Parameters:
    -----------
        step_size - number of epochs to wait between updates.
        gamma - reduction factor.
    """
    def __init__(self, step_size: int, gamma: float):
        self.step_size = step_size 
        self.gamma = gamma

    def get_last_lr(self):
        return [
            base_lr * numpy.power(self.gamma, int(self.last_epoch // self.step_size))
            for base_lr in self.base_lrs
        ]

class PolyLRScheduler(_LRScheduler):
    """
    Implementation of the Polynomial Learnin Rate 
    Scheduler.

    Parameters:
    -----------
        max_iter - maximum number of iterations during training.
        gamma - reduction factor.
    """
    def __init__(self, max_iter: int, gamma: float):
        self.gamma = gamma
        self.max_iter = max_iter

    def get_last_lr(self) -> typing.List[float]:
        return [
            base_lr * numpy.power((1 - self.last_epoch / self.max_iter), self.gamma)
            for base_lr in self.base_lrs 
        ]

class ExponentialLRScheduler(_LRScheduler):
    """
    Implementation of the Exponential Learning Rate
    Scheduler.
    
    Parameters:
    -----------
        max_iter - maximum number of epochs during training.
        gamma - decreasing factor
    """
    def __init__(self, max_iter: int, gamma: float):
        self.max_iter = max_iter
        self.gamma = gamma

    def get_last_lr(self) -> typing.List[float]:
        return [
            base_lr * numpy.exp(-self.gamma*self.last_epoch)
            for base_lr in self.base_lrs
        ]

class CosineAnnealingLRScheduler(_LRScheduler):
    """
    Implementation of the Cosine Annealing
    Learning Rate Scheduler.
    """
    def __init__(self, 
        eta_min: float, 
        eta_max: float, 
        T: float, 
        t_i: float, 
        t_c: float
    ):
        self.eta_min: float = eta_min 
        self.eta_max: float = eta_max
        self.T: float = T
        self.t_i: float = t_i
        self.t_c: float = t_c
    
    def get_last_lr(self) -> typing.List[float]:
        return [base_lr * (
            self.eta_min + (0.5 * (self.eta_max - self.eta_min) * ) for base_lr in self.base_lrs]

