from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.training.callbacks import base
import pathlib

@dataclass
class BaseNetworkConfig(object):
    """
    Implementation of the Network
    Configuration.

    Parameters:
    -----------
        network_path - (str) - path to the network (.pt, .pth) file.
    """
    network_path: pathlib.Path

@dataclass
class BaseOptimizerConfig(object):
    """
    Implementation of the optimization
    algorithm configuration.

    Parameters:
    -----------
        name: str - name of the optimizer
        lr: float - learning rate of the optimizer
        weight_decay: float - weight decay of the optimizer
        use_nesterov: float - use nesterov acceleration for the optimizer
    """
    name: str
    lr: float
    weight_decay: float = 0.0

@dataclass
class BaseLRSchedulerConfig(object):
    """
    Configuration for the LR Scheduler
    Parameters:
    ----------- 
        name - name of the LR Scheduler
    """
    name: str
    verbose: bool = False

@dataclass
class BaseCallbackConfig(object):
    """
    Implementation of the trainer
    callback configuration.
    Parameters:
    ----------
        callbacks - list of callbacks, that will be used
        during training
    """
    callbacks: typing.List[base.BaseCallback] = []

@dataclass
class BaseEarlyStoppingConfig(object):
    """
    Implementation of the Early Stopping
    Regularization configuration.
    
    Parameters:
    ----------
        patience - patience steps for the regularization
        min_diff - minimum float difference between two adjacent metric values
    """
    patience: int
    min_diff: float

@dataclass
class BaseTrainerConfig(object):
    """
    Base Implementation class
    for trainer configuration. It contains
    settings for neural network, optimization algorithm,
    lr scheduling algorithm and other important 
    """
    network_config: NetworkConfig
    optimizer_config: BaseOptimizerConfig
    callback_config: BaseCallbackConfig
    early_stopping_config: BaseEarlyStoppingConfig
    lr_scheduler_config: BaseLRSchedulerConfig = None
    batch_size: int
    max_epochs: int
