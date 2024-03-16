from src.training.trainers import base 
from src.training.contrastive_learning import sampler

from torch.utils.data import dataset
from torch.utils import data
from torch import nn

import torch
import typing

class ClassifierTrainer(base.BaseTrainer):
    """
    Training pipeline for MLP classification 
    network.
    
    Parameters:
    -----------
        network_config - configuration of the network.
        optimizer_config - configuration of the optimizer network.
        lr_scheduler_config - configuration of the LR scheduling network.
        max_epochs - maximum number of epochs to use during training.
        batch_size - 
    """
    def __init__(self, 
        batch_size: int, 
        max_epochs: int,
        network_config: typing.Dict,
        optimizer_config: typing.Dict,
        lr_scheduler_config: typing.Dict):
        super(ClassifierTrainer, self).__init__()

        self.network = self.configure_network(network_config)
        self.optimizer = self.configure_optimizer(optimizer_config)
        self.lr_scheduler = self.configure_lr_scheduler(lr_scheduler_config)
        self.max_epochs: int = max_epochs 
        self.batch_size: int = batch_size
        self.callbacks = self.configure_callbacks()

    def configure_optimizer(self, 
        network: nn.Module, 
        optimizer_config: typing.Dict
    ):
        pass

    def configure_lr_scheduler(self, 
        optimizer: nn.Module, 
        scheduler_config: typing.Dict
    ):
        pass 

    def configure_loader(self, dataset: dataset.Dataset, batch_size: int):
        if self.distributed:
            return data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True
            )
        else:
            return data.DataLoader(
                dataset=dataset,
                shuffle=True,
                batch_sampler=sampler.HardMiningContrastiveSampler(
                    video_data=(dataset.video_paths, dataset.video_labels),
                    textual_data=(dataset.text_paths, dataset.text_labels),
                    audio_data=(dataset.audio_paths, dataset.audio_labels)
                )
            )

    def configure_seed(self, base_seed: int):
        pass

    def train(self, train_dataset: dataset.Dataset):
        pass 
    
    def evaluate(self, validation_dataset: dataset.Dataset):
        pass
    
    def inference(self, input_imgs: torch.Tensor):
        pass
    



