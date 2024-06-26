from src.quantization.base import BaseQuantizer 
from torch.utils import data
from torch import nn
import torch
import logging 
import typing
import torch.ao.quantization

quan_logger = logging.getLogger(__name__)
handler = logging.FileHandler(filename='quantization_logs.log')
quan_logger.addHandler(handler)


class DynamicInferenceQuantizer(BaseQuantizer):
    """
    Module for dynamically quantizing parameters
    during network inference.
    """
    def __init__(self, network: nn.Module, quan_type: torch.dtype):
        self.quan_type = quan_type
        self.application_values: list = quan_type
        self.network = network

    def quantize(self) -> nn.Module:
        torch.quantization.quantize_dynamic(
            model=self.network,
            qconfig_spec={nn.Linear},
            dtype=self.quan_type
        )

class StaticNetworkQuantizer(object):
    """
    Base module for performing static quantization
    of the network.

    Parameters:
    -----------
        activation_observer_name - name of the activation quantization observer
        weight_observer_name - name of the weight quantization observer
        q_activation_type - dtype for activation observer
        q_weight_type - dtype for weight observer
        calibration_batch_size - batch size for calibration
        calibration_dataset - dataset to use for calibration
        inference_device - inference device to use for static quantization.

        NOTE:
            you won't be able to move quantized network
            from 'inference_device' to another device, without
            repeatedly retraining it. Be sure what device to use for 
            these purposes.
    """
    def __init__(self, 
        activation_observer_name: str,
        weight_observer_name: str,
        q_activation_type, 
        q_weight_type,
        calibration_batch_size: int,
        calibration_dataset: data.Dataset,
        inference_device: torch.DeviceObjType
    ):
        self.q_weight_type = q_weight_type 
        self.q_activation_type = q_activation_type 
        self.q_activation_type = q_activation_type
        self.q_weight_type = q_weight_type
        self.calibrator = NetworkCalibrator(inference_device=inference_device)
        self.calibration_dataset = calibration_dataset
        self.calibration_batch_size: int = calibration_batch_size
        self.activation_observer_name = activation_observer_name
        self.weight_observer_name = weight_observer_name

        if inference_device == "cuda":
            torch.backends.quantized.engine = "cudnn"

    def quantize(self, 
        input_model: nn.Module
    ):
        try:
            calibration_loader = self.calibrator.configure_calibration_loader(
                calibration_dataset=self.calibration_dataset,
                calibration_batch_size=self.calibration_batch_size,
                loader_workers=2
            )
            # perform calibration
            stat_network = self.calibrator.calibrate(
                input_model,
                loader=calibration_loader,
                activation_q_type=self.q_activation_type,
                weight_q_type=self.q_weight_type,
                weight_observer_name=self.weight_observer_name,
                activation_observer_name=self.activation_observer_name
            )

            if stat_network is None:
                raise RuntimeError("failed to calibrate network")

            quantized_model = torch.quantization.convert(stat_network)
            return quantized_model 

        except(Exception) as err:
            quan_logger.error(err)
            raise err

class NetworkCalibrator(object):
    """
    Base module for performing
    calibration for static post-training
    quantization.
    """
    def __init__(self, inference_device: torch.DeviceObjType):
        self.inf_device = inference_device
        
    def configure_calibration_loader(self, 
        calibration_dataset: data.Dataset,
        calibration_batch_size: int,
        loader_workers: int = 0
    ):
        """
        Configures data loader for
        performing calibration.
        """
        return data.DataLoader(
            dataset=calibration_dataset,
            batch_size=calibration_batch_size,
            shuffle=True,
            num_workers=loader_workers,
        )

    @staticmethod
    def configure_observer(observer_name: str):
        """
        Configures observer method for defining
        quantization range.
        
        Parameters:
        -----------
            observer_name (str) - name of the observer class to use.
        """
        if observer_name.lower() == "percentile":
            return torch.ao.quantization.observer.PercentileObserver

        if observer_name.lower() == "minmax":
            return torch.ao.quantization.observer.MinMaxObserver
        
        if observer_name.lower() == "histogram":
            return torch.ao.quantization.observer.HistogramObserver
        
        if observer_name.lower() == "moving_minmax":
            return torch.ao.quantization.MovingMinMaxObserver

    def calibrate(self, 
        network: nn.Module, 
        loader: data.DataLoader,
        weight_observer_name: typing.Literal['percentile', 'minmax', 'histogram'],
        activation_observer_name: typing.Literal['percentile', 'minmax', 'histogram'],
        activation_q_type,
        weight_q_type,
    ) -> typing.Union[nn.Module, None]:
        """
        Calibrates given network
        for finding optimal quantization
        parameters.
        
        NOTE:
            loader should contain a dataset,
            which is originally derived from
            the training set.
        """
        network.eval()
        try:
            # defining weight and activation observers
            weight_observer = self.configure_observer(observer_name=weight_observer_name)
            activation_observer = self.configure_observer(observer_name=activation_observer_name)

            # Specify the quantization configuration
            qconfig = torch.ao.quantization.QConfig(
                activation=activation_observer.with_args(
                    dtype=activation_q_type,
                    qscheme=torch.per_tensor_affine
                ),
                weight=weight_observer.with_args(
                    dtype=weight_q_type,
                    qscheme=torch.per_tensor_symmetric
                )
            )
            # Apply the quantization configuration to the model
            network.qconfig = qconfig
            stat_network = torch.ao.quantization.prepare(network)

            # performing calibration 
            for images, _ in loader:
                dev_images = images.to(self.inf_device)
                stat_network.to(self.inf_device).forward(dev_images).cpu()

            return stat_network

        except(Exception) as err:
            quan_logger.error(err)
            raise RuntimeError("failed to calibrate network, check logs.")