from torch import nn 
from torchvision.models.video import S3D
import torch 

PRETRAINED_VIDEO_ENCODER_URL = "https://pytorch.s3.amazonaws.com/models/multimodal/mugen/video_encoder-weights-b0e27f13.pth"

class VideoEncoder(nn.Module):
    """
    Encodes videos to the last layer before
    passing to the S3D network for further processing.
    """
    def __init__(self ):
        self.model = S3D()
        self.out_dim = self.model.classifier[1].in_channels
        self.model.classifier = nn.Identity(self.out_dim)
    
    def forward(self, input_imgs: torch.Tensor):
        if input_imgs.shape[-1] == 3:
            raise ValueError()
        return self.model(input_imgs)