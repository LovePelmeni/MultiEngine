from abc import abstractmethod 
from src.training.callbacks import base

from torch import optim
from torch.optim import lr_scheduler
from torch import device

from torch.utils import data
from torch.distributed.optim import zero_redundancy_optimizer as zero

from src.training.trainers.base_config import (
    BaseTrainerConfig,
    BaseOptimizerConfig,
    BaseLRSchedulerConfig,
    BaseEarlyStoppingConfig
)

import typing
from torch import nn

class AbstractBaseTrainer(base.TrainerCallbackMixin):
    """
    Abstraction class, representing base
    trainer pipeline for training models
    """
    @abstractmethod 
    def configure_optimizer(self, **kwargs):
        """
        Configures optimization algorithm to
        guide training updates.
        
        Warning:
            this is just empty shell, which is reinitialized 
            in other classes
        """

    @abstractmethod
    def configure_lr_scheduler(self, **kwargs):
        """
        Configures LR Scheduling Algorithm to dynamically
        change learning rate during training
        Warning:
            this is just empty shell, which is reinitialized
            in other classes.
        """
    
    @abstractmethod
    def configure_loader(self, **kwargs):
        """
        Configures data loader for training / validation
        stage.
        Warning:
            this is just empty shell, which is reinitilized in 
            other classes.
        """

    @abstractmethod
    def configure_early_stopping(self, **kwargs):
        """
        Configures early stopping regularization 
        Warning:
            this is just empty shell, which is reinitialized
            in other classes.
        """
    
    @abstractmethod
    def configure_seed(self, **kwargs):
        """
        Configures randomness seed to set
        deterministic status for the experiment.
        Warning:
            this is just empty shell, which is reinitiliazed
            in other classes.
        """

    @abstractmethod
    def load_metrics(self, **kwargs):
        """
        Loades evaluation metrics for 
        validation of the network.
        Warning:
            this is just empty shell, which is reinitialize
            in other classes.
        """

    @abstractmethod
    def load_losses(self, **kwargs):
        """
        Loades loss functions for training
        network.
        Warning:
            this is just empty shell, which is reinitialized
            in other classes.
        """

    @abstractmethod
    def configure_callbacks(self, **kwargs):
        """
        Configures performance tracking callbacks
        to track performance metrics during training of the network.
        Warning:
            this is just empty shell, which is reinitilized in
            other classes.
        """

    @abstractmethod
    def inference(self, **kwargs):
        """
        Performs inference of the network
        Warning:
            this is just empty shell, which is reinitilized in
            other classes.
        """
    
    @abstractmethod
    def train(self, **kwargs):
        """
        Performs training of the network step by step.
        Warning:
            this is just empty shell, which is reinitilized in
            other classes.
        """

    @abstractmethod
    def stop(self, **kwargs):
        """
        Performs stop of the training process.
        Warning:
            this is just empty shell, which is reinitialized in
            other classes.
        """

class BaseTrainer(AbstractBaseTrainer):
    """
    Implementation of the Base Training class,
    which abstracts away general functionality required
    for training network.
    """
    def __init__(self, distributed: bool = False, **kwargs):
        self.distributed = distributed

        # setting up additional attributes
        for attr_name, attr_value in kwargs.items():
            if not hasattr(self, attr_name):
                setattr(self, attr_name, attr_value)

    def configure_optimizer(self, 
        network: nn.Module, 
        optimizer_config: BaseOptimizerConfig) -> nn.Module:

        optimizer_name = optimizer_config.get("name")
        learning_rate = optimizer_config.get("learning_rate")
        weight_decay = optimizer_config.get("weight_decay", None)
        use_nesterov = optimizer_config.get("nesterov", False)

        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                params=network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        elif optimizer_name.lower() == 'adamax':
            optimizer = optim.Adamax(
                params=network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(
                params=network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                params=network.parameters(),
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                nesterov=use_nesterov
            )
        else:
            raise NotImplemented()

        if (self.distributed == True):
            optimizer = zero.ZeroRedundancyOptimizer(
                params=network.parameters(),
                optimizer_class=optimizer_name,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        return optimizer

    def configure_loader(self, 
        dataset: data.Dataset, 
        num_workers: int,
        batch_size: int, 
        distributed: bool = False,
        num_replicas: int = 1) -> data.DataLoader:
        """
        Configures data loader for
        training / validation phase.
        """
        if distributed:
            return data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=num_workers,
                sampler=data.DistributedSampler(
                    dataset=dataset, 
                    num_replicas=num_replicas
                ),
            )
        else:
            return data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

    def configure_lr_scheduler(self, 
        optimizer: nn.Module, 
        lr_scheduler_config: BaseLRSchedulerConfig) -> nn.Module:
        """
        Supports:
            'poly', 'step', 'multistep', 'exp';
        """
        name = lr_scheduler_config.get("name")
        verbose = lr_scheduler_config.get("verbose", False)
        gamma = lr_scheduler_config.get("gamma")
        total_iters = lr_scheduler_config.get("total_iters")

        if name == 'poly':
            return lr_scheduler.PolynomialLR(
                optimizer=optimizer,
                total_iters=total_iters,
                power=gamma,
                verbose=verbose
            )
        if name == 'step':
            step_size = lr_scheduler_config.get("step_size")
            return lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=step_size,
                gamma=gamma,
                verbose=verbose
            )

        if name == 'multistep':
            steps = lr_scheduler_config.get("steps")
            return lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=steps,
                gamma=gamma,
                verbose=verbose
            )

        if name == 'exp':
            return lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=gamma,
                verbose=verbose
            )

    def configure_device(self, device_name: str):
        return device(device_name)

    def configure_early_stopping(self, early_stopping_config: BaseEarlyStoppingConfig):
        pass

    def configure_network(self, 
        network_config: BaseNetworkConfig, 
        device_ids: typing.List[torch.device],
        output_device: str = 'cpu') -> nn.Module:
        try:
            network = torch.load(network_config.network_path)
        except(Exception) as err:
            raise RuntimeError(err)
        
        if (self.distributed == True):
            conf_network = DDP(
                network, 
                device_ids=device_ids, 
                output_device=output_device
            )
        else:
            devices = ','.join(device_ids)
            conf_network = network.to(device=devices)
        return conf_network
    