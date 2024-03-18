from torch import nn
import logging 
import typing
import torch

logger = logging.getLogger("attention_fusion_logger")
handler = logging.FileHandler(filename='attention_fusion_logs.log')
logger.addHandler(handler)

class AttentionFusion(nn.Module):
    """
    Implementation of the Attention-based
    Fusion layer for aligning multimodal output 
    data.
    
    Parameters:
    -----------
        1. channel_to_encoder_dim - dictionary
            which maps encoder to it's number of channels.
            Example:
                {'encoder1': 1024, 'encoder2': 512},
    """
    def __init__(self, 
        channel_to_encoder_dim: typing.Dict[str, int], 
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

    def load_fusion_state(self, fusion_state: nn.Module):
        self.attention = self.attention.load_state_dict(state_dict=fusion_state)
        
    def forward(self, embeddings: typing.Dict[str, torch.Tensor]):
        """
        NOTE:
            it is implied, that embeddings are already
            projected into the same latent space and
            their length is equal.
        """
        emb_projections = list(embeddings.values())
        concat_embs = torch.cat(
            [embeddings[key] for key in sorted(embeddings.keys())],
            dim=-1
        )
        attention_weights = self.attention(concat_embs)
        output_projections: typing.List[torch.Tensor] = []
 
        for emb in range(len(attention_weights)):
            output_proj = attention_weights[:, emb].unsqueeze(-1) * emb_projections[emb]
            output_projections.append(output_proj)
        
        weighted_sum_tensor = torch.sum(torch.stack(output_projections), dim=0)
        return weighted_sum_tensor




