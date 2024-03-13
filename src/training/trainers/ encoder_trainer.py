from src.training.trainers import base
from src.training.datasets import base as data_base
from src.training.callbacks import (
    checkpoints,
    devices,
    logistics,
    early_stopping
)
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch import nn, Tensor
import torch
import typing
import logging 
from torch import optim 
from torch.optim import lr_scheduler 
from torch import device
import pathlib
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EncoderTrainer(base.BaseTrainer):
    """
    Trainer pipeline for 
    Embedding Generation Network.

    Parameters:
    -----------
        network_config -  sffdsd
    """
    def __init__(self, 
        network_config: typing.Dict,
        optimizer_config: typing.Dict,
        lr_scheduler_config: typing.Dict,
        snapshot_config: typing.Dict,
        early_stopping_config: typing.Dict,
        loader_config: typing.Dict,
        inference_device: str,
        distributed: bool = False,
        seed: int = None
    ):
        self.device = self.configure_device(
            inference_device=inference_device,
            distributed=distributed
        )
        self.loader_config = loader_config
        self.snapshot_config = snapshot_config 
        self.early_stopping_config = early_stopping_config
        self.network = self.configure_network(**network_config)
        self.optimizer = self.configure_optimizer(network=self.network, **optimizer_config)
        self.lr_scheduler = self.configure_lr_scheduler(optimizer=self.optimizer, **lr_scheduler_config)
        self.losses = self.load_losses()
        self.metrics = self.load_metrics()
        self.configure_seed(seed=seed)
        self.stop_flag = False

    def configure_optimizer(self, network: nn.Module, optimizer_config: typing.Dict):

        name = optimizer_config.get("name")
        learning_rate = optimizer_config.get("learning_rate")
        weight_decay = optimizer_config.get("weight_decay", None)
        use_nesterov = optimizer_config.get("nesterov", False)

        if name.lower() == 'adam':
            return optim.Adam(
                params=network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        if name.lower() == 'adamax':
            return optim.Adamax(
                params=network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        if name.lower() == 'adadelta':
            return optim.Adadelta()
        
        if name.lower() == 'rmsprop':
            return optim.RMSprop(
                params=network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        if name.lower() == 'sgd':
            return optim.SGD(
                params=network.parameters(),
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                nesterov=use_nesterov
            )
        else:
            raise NotImplemented()

    def configure_loader(self, 
        dataset: data_base.BaseDataset, 
        num_workers: int,
        batch_size: int, 
        distributed: bool = False,
        num_replicas: int = 1
    ):
        if distributed:
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                pin_memory=True,
                shuffle=False,
                num_workers=num_workers,
                sampler=DistributedSampler(
                    dataset=dataset, 
                    num_replicas=num_replicas
                ),
            )
        else:
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

    def configure_lr_scheduler(self, optimizer: nn.Module, lr_scheduler_config: typing.Dict):
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

    def configure_seed(self, seed: int):
        torch.manual_seed(seed=seed)

    def configure_callbacks(self, base_log_dir: typing.Union[str, pathlib.Path]):

        report_log_dir = os.path.join(base_log_dir, "reports")
        cpu_log_dir = os.path.join(base_log_dir, "cpu")
        gpu_log_dir = os.path.join(base_log_dir, "gpu")

        snapshot_log_dir = os.path.join(base_log_dir, "snapshots")
        snapshot_ext = self.snapshot_config.get("snapshot_ext")
        save_every = self.snapshot_config.get("save_every")

        min_diff = self.early_stopping_config.get("min_diff")
        patience = self.early_stopping_config.get("patience")
        validation_dataset = self.early_stopping_config.get("validation_dataset")
        
        self.callbacks = [
            logistics.LogisticsCallback(log_dir=report_log_dir),
            devices.CPUInferenceCallback(log_dir=cpu_log_dir),
            devices.GPUInferenceCallback(log_dir=gpu_log_dir),
            checkpoints.SnapshotCallback(
                snapshot_ext=snapshot_ext, 
                save_every=save_every, 
                log_dir=snapshot_log_dir
            ),
            early_stopping.EarlyStoppingCallback(
                min_diff=min_diff,
                patience=patience,
                validation_dataset=validation_dataset
            )
        ]

    def train(self, train_dataset: data_base.BaseDataset):

        self.on_init_start()
        self.network.train()
        loader = self.configure_loader(train_dataset, **self.loader_config)
        self.on_init_end()
        
        curr_loss = float('inf')

        for epoch in range(self.max_epochs):

            for images, labels in tqdm(
                iterable=loader, 
                desc='epoch: %s; loss: %s;' % (epoch, curr_loss)
            ):

                self.on_train_batch_start()
                device_data = images.float().to(self.device)
                predictions = self.network.to(self.device).forward(device_data).cpu()
                self.on_train_batch_end()

                for loss in self.losses:
                    loss_val = loss(predictions, labels)
                    loss_val.backward()

                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch=epoch)

            self.on_validation_start(trainer=self)
            self.on_validation_end()

            self.on_train_epoch_end()
            if (self.stop_flag == True):
                break

    def evaluate(self, validation_dataset: data_base.BaseDataset) -> typing.List[float]:

        self.network.eval()
        loader = self.configure_loader(validation_dataset)

        with torch.no_grad():

            output_predictions = []
            output_labels = []
            metrics = []

            for images, labels in loader:
                predictions = self.network.forward(images).cpu()
                output_predictions.append(predictions)
                output_labels.append(labels)

            for metric in self.metrics:
                metric_value = metric(
                    torch.as_tensor(predictions), 
                    torch.as_tensor(output_labels)
                )
                metrics.append(metric_value)
        return metrics

    def inference(self, input_images: typing.List[Tensor]):
        preds = []
        for input_image in input_images:
            prediction = self.network.forward(input_image)
            preds.append(prediction)
        return preds