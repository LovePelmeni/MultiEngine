from pytorch_grad_cam.grad_cam import GradCAM
from torch import nn
import typing
import torch
import cv2
import matplotlib.pyplot as plt

class CNNExplainer(object):
    """
    Base module for interpreting CNN-based
    embedding generation networks.

    Parameters:
    -----------
        cnn_encoder - CNN-based encoder network to interpret
        target_layers - last convolutional layer of the CNN encoder.
    """
    def __init__(self, cnn_encoder: nn.Module, target_layers: typing.List[nn.Module]):
        self.cnn_encoder = cnn_encoder
        self.cam_interpreter = GradCAM(
            model=cnn_encoder, 
            target_layers=target_layers
        )

    def explain(self, 
        input_images: typing.List[torch.Tensor], 
        target_labels: typing.List[int]
    ):
        """
        Generates qualitative feature map,
        that describes important regions on the image
        for a series of input images.
        
        Parameters:
        -----------
            input_images - list of (height, width, channel) torch.Tensor RGB or (height, width) grayscale images.
            target_labels - list of corresponding target labels.
        """
        _, ax = plt.subplots(ncols=2, nrows=len(input_images))
        curr_img = 0

        for input_img, input_label in zip(input_images, target_labels):
            img_height, img_width = input_img.shape
            predicted_map = self.cam_interpreter.forward(
                input_tensor=input_img.permute(2, 0, 1).unsqueeze(0), 
                targets=[input_label]
            )
            resized_map = cv2.resize(
                predicted_map, 
                (img_height, img_width), 
                interpolation=cv2.INTER_LINEAR
            )
            colored_map = cv2.applyColorMap(resized_map, cv2.COLORMAP_JET)
            mixed_map = cv2.addWeighted(input_img.numpy(), 0.7, colored_map, 0.3)
            ax[curr_img, 0].imshow(input_img)
            ax[curr_img, 1].imshow(mixed_map)
            curr_img += 1

