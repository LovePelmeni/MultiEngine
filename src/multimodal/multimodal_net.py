from torch import nn
from src.multimodal.image_encoder import ImageEncoder
from src.multimodal.desc_encoder import DescriptionEncoder
from src.multimodal.title_encoder import TitleEncoder 
from src.multimodal.fusions.attention_fusion import AttentionFusion
import torch 

class MultimodalNetwork(nn.Module):
    """
    Multimodal network for handling 
    data from multiple sources (modalities)
    including video, text and audio emotion
    information.

    Parameters:
    -----------
        video_encoder - network for processing video and generating video embeddings
        text_encoder - network for processing text units and generating text embeddings
        audio_encoder - network for processing audio sequences and generating audio embeddings
        fusion_layer - layer for fusing embeddings and aligning them accordingly.
    """
    def __init__(self, 
        image_encoder: ImageEncoder, 
        desc_encoder: DescriptionEncoder, 
        title_encoder: TitleEncoder,
        fusion_layer: AttentionFusion
    ):
        super(MultimodalNetwork, self).__init__()

        self.image_encoder = image_encoder
        self.desc_encoder = text_encoder
        self.title_encoder = title_encoder
        self.fusion_layer = fusion_layer

    def forward(self, 
        input_image: torch.Tensor = None, # tensor of images
        input_desc: torch.Tensor = None, # tensor of strings
        input_title: torch.Tensor = None # tensor of strings
    ) -> torch.Tensor:
        """
        NOTE:
            you should pass modalities to fusion
            in the same order, as was used during training.
        """
        fused_embs = self.fusion_layer(
            modalities=[
                input_image, 
                input_desc, 
                input_title
            ],
            classifiers=[
                self.image_encoder, 
                self.desc_encoder, 
                self.title_encoder
            ]
        )
        return fused_embs




