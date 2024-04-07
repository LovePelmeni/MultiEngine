from src.training.callbacks import base 
from torch.utils.tensorboard.writer import SummaryWriter
import pathlib
from torch import nn
import typing
import torch

class NetworkMonitoringCallback(base.BaseCallback):
    """
    Base callback for monitoring health of the network
    during training phase.
    
    It tracks following information:
        1. change flow of layer weights.
        2. loss value for each epoch.
        3. 
    """
    def __init__(self, 
        log_dir: typing.Union[str, pathlib.Path],
        weight_param_tag: str = 'weights', 
        bias_param_tag: str = 'biases',
    ):
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=120)
        self.weight_log_tag = weight_param_tag 
        self.bias_log_tag = bias_param_tag

    def _track_loss(self, loss_value: float, global_step: int):
        self.writer.add_scalar(
            tag="train_loss", 
            scalar_value=loss_value, 
            global_step=global_step
        )

    def _track_evaluation_metric(self, eval_value: float, global_step: int):
        self.writer.add_scalar(
            tag="eval_metric", 
            scalar_value=eval_value, 
            global_step=global_step
        )

    def _track_learning_rate(self, value: float, global_step: int):
        self.writer.add_scalar(
            tag="learning_rate", 
            scalar_value=value, 
            global_step=global_step
        )

    def _track_epoch_time(self, value: float, global_step: int):
        self.writer.add_scalar(
            tag="epoch_time", 
            scalar_value=value, 
            global_step=global_step
        )

    def _track_conv2d_params(self, 
        conv_weights: torch.Tensor, 
        tag: str, 
        global_step: int
    ):
        weights_shape = conv_weights.shape
        num_kernels = weights_shape[0]
        for k in range(num_kernels):
            flattened_weights = conv_weights[k].flatten()
            self.writer.add_histogram(
                tag, flattened_weights, 
                global_step=global_step, bins='tensorflow'
            )

    def _track_linear_params(self, 
        tag: int, global_step: int, 
        linear_weights: torch.Tensor
    ):
        self.writer.add_histogram(
            tag=tag, 
            global_step=global_step,
            values=linear_weights.numpy()
        )

    def _track_layer_parameters(self, network: nn.Module, global_step: int):
        """
        Track disribution of network weights and biases
        over epochs.

        Parameters:
        -----------
            network - nn.Module neural network, currently in the training mode
        """
        for param_name, param in network.named_parameters():

            if (param.requires_grad == True):

                if ('weight' in param_name):
                    tag_name = self.weight_log_tag

                elif ('bias' in param_name):
                    tag_name = self.bias_log_tag

                self.writer.add_histogram(
                    tag="%s/%s" % (param_name, tag_name),
                    values=param.clone().cpu().data.numpy(),
                    global_step=global_step
                )

    def on_train_epoch_end(self, **kwargs):

        loss_value = kwargs.get("train_loss")
        eval_value = kwargs.get("eval_value")
        network = kwargs.get("network")
        global_step = kwargs.get("global_step")
        epoch_time = kwargs.get("epoch_time")
        learning_rate = kwargs.get("learning_rate")

        self._track_layer_parameters(network=network, global_step=global_step)
        self._track_loss(loss_value=loss_value, global_step=global_step)
        self._track_evaluation_metric(loss_value=eval_value, global_step=global_step)
        self._track_learning_rate(value=learning_rate, global_step=global_step)
        self._track_epoch_time(value=epoch_time, global_step=global_step)