from abc import abstractmethod 
from src.training.callbacks import base

class BaseTrainer(base.TrainerCallbackMixin):
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




