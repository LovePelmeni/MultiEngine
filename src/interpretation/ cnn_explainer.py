from pytorch_grad_cam.grad_cam import GradCAM
from src.interpretation import base
import torch
from torch import nn
import typing
import cv2
import matplotlib.pyplot as plt
import numpy


class CNNExplainer(base.BaseExplainer):
    """
    Base module for interpreting CNN-based
    network architectures
    """
    def __init__(self, 
        last_network_layer: nn.Module,
        network: nn.Module, 
        inference_device: typing.Literal['cpu', 'cuda', 'm'],
    ):
        self.last_network_layer = last_network_layer
        self.network = network.to(inference_device)
        self.inference_device = torch.device(inference_device)
        self.cam = GradCAM(
            model=self.network,
            target_layers=[self.last_network_layer],
            reshape_transform=None,
        )

    @staticmethod
    def visualize_cam_map(
        input_image: torch.Tensor, 
        input_map: torch.Tensor,
        c1: float,
        c2: float
    ):
        """
        Fuses image and grad-cam generated map
        to provide a high quality qualititative interpretation
        of the network prediction.
        
        Parameters:
        -----------
            input_image - numpy.ndarray image 
            input_map - numpy.ndarray grad-cam map, same shape as input_image
        """
        _, ax = plt.subplots(ncols=2, nrows=1)
        heatmap = cv2.applyColorMap(input_map.astype(numpy.uint8), colormap=cv2.COLORMAP_JET)
        weighted_sum = cv2.addWeighted(input_image, alpha=c1, src2=heatmap, beta=c2)
        ax[0].imshow(input_image)
        ax[1].imshow(weighted_sum)
    
    def explain(self, input_image: torch.Tensor, target_embedding: torch.Tensor):
        predicted_map =self.cam.forward(
            input_tensor=input_image, 
            targets=target_embedding.to(self.inference_device), 
            eigen_smooth=False
        )
        img_height, img_width = input_image.shape[:2]
        resized_map = cv2.resize(
            src=predicted_map,
            dsize=(img_height, img_width), 
            interpolation=cv2.INTER_LINEAR
        )
        self.visualize_cam_map(input_image.numpy(), resized_map, c1=0.6, c2=0.4)
         