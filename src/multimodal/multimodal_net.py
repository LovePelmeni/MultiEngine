from torch import nn
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
        image_encoder: nn.Module, 
        text_encoder: nn.Module, 
        fusion_layer: nn.Module
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

