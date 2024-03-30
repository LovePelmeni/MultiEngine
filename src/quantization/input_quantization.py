from src.quantization import base
import torch
import numpy

class TensorInputQuantizer(base.BaseQuantizer):
    """
    Module for quantizing input data
    parameters into lower precision per tensor.
    
    Parameters:
    -----------
        quan_type - type to use for quantization (torch.qint8, ...)
        trained_observer - observer, trained on the similar image dataset.
    """
    def __init__(self, 
        quan_type: torch.dtype, 
        trained_observer: torch.quantization.ObserverBase
    ):
        self.quan_type = quan_type
        self.observer = trained_observer

    def compute_quantization_logistics(self, input_img: torch.Tensor):
        scale, zero_point = self.observer.calculate_qparams()
        return scale, zero_point

    def quantize(self, input_img: torch.Tensor):
        scale, zero_point = self.compute_quantization_logistics(input_img)
        return torch.quantize_per_tensor(
            input=input_img,
            scale=scale,
            zero_point=zero_point,
            dtype=self.quan_type
        )

class ChannelInputQuantizer(base.BaseQuantizer):
    """
    Module for quantizing parameters
    of the input per channel independently.
    
    Parameters:
    -----------
        quan_type - type to use for quantization (torch.qint8, ....)
        trained_observer - observer, pretrained on the similar image dataset.
    """
    def __init__(
        self,
        quan_type: torch.dtype,
        trained_observer: torch.quantization.ObserverBase
    ):
        self.quan_type = quan_type
        self.observer = trained_observer

    def compute_quantization_logistics(self, input_img: torch.Tensor):
        scales, zero_points = self.observer.compute_qparams(input_img)
        return scales, zero_points
    
    def quantize(self, input_img: torch.Tensor):
        return torch.quantize_per_channel(
            input=input_img,
            zero_points=self.zero_points,
            scales=self.scales,
            dtype=self.quan_type
        )


class VideoInputQuantizer(object):
    """
    Base module for quantizing video streams
    for Video-based classification.
    
    Parameters:
    ----------- 
        input_observer - observer to calculate statistics,
         should be pre-trained on video relative dataset.
        frame_window_size - number of frames to use during one quantization step.
    """
    def __init__(self, 
        quantization_type: numpy.dtype,
        frame_window_size: int,
        average_constant_rate: float
    ):
        self.input_observer = torch.quantization.MovingAverageMinMaxObserver(
            average_constant=average_constant_rate,
            dtype=quantization_type,
        )
        self.quantization_type = quantization_type
        self.frame_window_size = frame_window_size
    
    def quantize(self, input_video: numpy.ndarray):

        total_frames = input_video.shape[0]
        quantized_video = numpy.empty(shape=input_video.shape)

        for step in range(total_frames // self.frame_window_size):

            start = step*self.frame_window_size 
            end = start + self.frame_window_size

            batch_frames = input_video[start:end]
            self.input_observer(batch_frames)

            self.input_observer(batch_frames)
            zero_point, scale = self.input_observer.compute_qparams()
            
            quantized_frames = torch.quantize_per_tensor(
                input=batch_frames, 
                scale=scale,
                zero_point=zero_point
            )
            quantized_video[start:end] = quantized_frames
        return quantized_video