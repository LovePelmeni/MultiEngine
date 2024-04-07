from torch import nn
import torch
from src.multimodal.image_encoder import ImageEncoder
from src.multimodal.text_encoder import TextEncoder 
from src.multimodal.fusions.attention_fusion import AttentionFusion

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
        text_encoder: TextEncoder, 
        fusion_layer: AttentionFusion
    ):
        super(MultimodalNetwork, self).__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_layer = fusion_layer

    def forward(self, 
        input_image: torch.Tensor = None, 
        input_text: torch.Tensor = None,
    ) -> torch.Tensor:
        fused_embs = self.fusion_layer(
            modalities=[input_image, input_text],
            classifiers=[self.image_encoder, self.text_encoder]
        )
        return fused_embs

