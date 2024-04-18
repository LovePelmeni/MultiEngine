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
        fusion_layer: AttentionFusion,
        inference_device: torch.DeviceObjType
    ):
        super(MultimodalNetwork, self).__init__()

        self.image_encoder = image_encoder.to(inference_device)
        self.desc_encoder = text_encoder.to(inference_device)
        self.title_encoder = title_encoder.to(inference_device)
        self.fusion_layer = fusion_layer.to(inference_device)
        
        for enc in [
            self.image_encoder, 
            self.desc_encoder, 
            self.title_encoder, 
            self.fusion_layer]:
            enc.eval()

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
        image_emb = self.image_encoder(input_image.float())
        desc_emb = self.desc_encoder(input_desc.float())
        title_emb = self.title_encoder(input_title.float())

        fused_embs = self.fusion_layer(
            modalities=[
                image_emb, 
                desc_emb, 
                title_emb
            ],
            classifiers=[
                self.image_encoder, 
                self.desc_encoder, 
                self.title_encoder
            ]
        )
        return fused_embs

class Applicati