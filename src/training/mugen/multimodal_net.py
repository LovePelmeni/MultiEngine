from torch import nn
from src.training.mugen import projection
from src.training.classifiers import classifiers
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
        video_encoder: nn.Module, 
        text_encoder: nn.Module, 
        audio_encoder: nn.Module,
        fusion_layer: nn.Module,
        embedding_length: int,
        output_classes: int,
    ):
        super(MultimodalNetwork, self).__init__()

        self.video_encoder = nn.Sequential(
            video_encoder,
            projection.ProjectionLayer(
                in_dim=video_encoder.out_dim, 
                out_dim=embedding_length
            )
        )
        self.text_encoder = nn.Sequential(
            text_encoder,
            projection.ProjectionLayer(
                in_dim=text_encoder.out_dim,
                out_dim=embedding_length
            )
        )
        self.audio_encoder = nn.Sequential(
            audio_encoder,
            projection.ProjectionLayer(
                in_dim=audio_encoder.out_dim,
                out_dim=embedding_length
            )
        )
        self.fusion_layer = fusion_layer
        self.classifier = classifiers.MultiLayerPerceptronClassifier(
            embedding_length=embedding_length,
            output_classes=output_classes,
        )
    
    def forward(self, 
        input_video: torch.Tensor = None, 
        input_text: torch.Tensor = None,
        input_audio: torch.Tensor = None
    ) -> torch.Tensor:
        fused_embs = self.fusion_layer(
            modalities=[input_video, input_text, input_audio],
            classifiers=[self.video_encoder, self.text_encoder, self.audio_encoder]
        )
        return fused_embs

    

