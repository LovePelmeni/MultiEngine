from torch import nn
import logging 
import typing
import torch

logger = logging.getLogger("attention_fusion_logger")
handler = logging.FileHandler(filename='attention_fusion_logs.log')
logger.addHandler(handler)

class AttentionFusion(nn.Module):
    """
    Implementation of the vanilla Attention-based
    Fusion for aligning multimodal output 
    data.
    
    Parameters:
    -----------
        1. channel_to_encoder_dim - list of
        the following format: [1, 2, 3], where 1, 2, 3 -> number
        of output dimensions for each encoder.

        2. fusion_weights (nn.Module) - in case 
        model has been pretrained and we want to load it's state
    """
    def __init__(self, 
        channel_to_encoder_dim: typing.List, 
        fusion_weights: nn.Module = None
    ):
        super(AttentionFusion, self).__init__()
        self.channel_to_encoder_dim = channel_to_encoder_dim

        # this linear layer corresponds to weights, that should be learned
        # during training. Each weight corresponds to a modality.
        attn_in_dim = sum(self.channel_to_encoder_dim.values())

        self.attention = nn.Sequential(
            nn.Linear(
                in_features=attn_in_dim, 
                out_features=len(channel_to_encoder_dim),
                bias=True
            ),
            nn.Softmax(-1),
        )

        if fusion_weights is not None:
            self.load_fusion_state(fusion_state=fusion_weights)
        
        self.attention.eval()

    def load_fusion_state(self, fusion_state: nn.Module):
        self.attention = self.attention.load_state_dict(state_dict=fusion_state)
        
    def forward(self, embeddings: typing.List[torch.Tensor]):
        """
        NOTE:
            it is implied, that embeddings are already
            projected into the same latent space and
            their length is equal.
        """
        concat_embs = torch.cat(embeddings, dim=-1)
        attention_weights = self.attention(concat_embs)
        output_projections: typing.List[torch.Tensor] = []
 
        for emb in range(len(attention_weights)):
            output_proj = attention_weights[:, emb].unsqueeze(-1) * embeddings[emb]
            output_projections.append(output_proj)
        
        weighted_sum_tensor = torch.sum(
            torch.stack(output_projections), 
            dim=0
        )
        return weighted_sum_tensor


class VisualDotProductAttentionFusion(nn.Module):
    """
    Attention-based fusion, what uses attention mechanism
    called "Visual dot product Attention".
    
    Paper: https://paperswithcode.com/method/dot-product-attention

    """
    def __init__(self, 
        channels_to_dim: typing.Dict[str, int], 
        fusion_weights: nn.Module = None
    ):
        super(VisualDotProductAttnetionFusion, self).__init__()

class CoAttentionFusion(nn.Module):
    """
    Attention-based fusion, that uses attention mechanism
    called "Co-Attention".
    
    Paper: https://www.researchgate.net/figure/The-structure-of-co-attention-mechanism_fig3_343606404

    """
    def __init__(self, 
        channels_to_dim: typing.Dict[str, int],
        fusion_weights: nn.Module = None
    ):
        super(CoAttentionFusion, self).__init__()

class StackedVisualAttentionFusion(nn.Module):
    """
    Attention-based fusion, that uses attention mechanism 
    called "Stacked Visual Attention".

    Paper: https://www.researchgate.net/publication/343842731_Stack-VS_Stacked_Visual-Semantic_Attention_for_Image_Caption_Generation

    """
    def __init__(self, 
        channels_to_dim: typing.Dict[str, int],
        fusion_weights: nn.Module = None
    ):
        super(StackedVisualAttentionFusion, self).__init__()