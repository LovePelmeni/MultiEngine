from src.training.mugen import (
    video_encoder,
    face_detector,
)

from torch import nn


class VideoCLIP(nn.Module):
    """
    Class validation network
    container string value.

    Parameters:
    -----------
    
    """
    def __init__(self, 
        text_pretrained_encoder: nn.Module, 
        video_pretrained_encoder: nn.Module,
        pretrained_face_detector: nn.Module,
    ):
        self.text_encoder = text_pretrained_encoder 
        self.video_encoder = video_pretrained_encoder
        self.pretrained_face_detector = pretrained_face_detector 

    def forward(self,):
        pass 

