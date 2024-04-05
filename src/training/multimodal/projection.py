from torch import nn 
import torch

class ProjectionLayer(nn.Module):
    """
    Projects embeddings from modality
    size to a fixed dimension.

    Parameters:
    ----------
        in_dim - dimension size of modality
        out_dim - dimension size of the linear projection to create
        dropout_prob - regularization prob fo dropout reg.
    """
    def __init__(self, in_dim: int, out_dim: int = 256, dropout_prob: float = 0.1):
        super(ProjectionLayer, self).__init__()

        self.dense1 = nn.Linear(
            in_features=in_dim, 
            out_features=out_dim, 
            bias=False
        )
        self.gelu = nn.GeLU()
        self.dense2 = nn.Linear(
            in_features=out_dim,
            out_features=out_dim,
            bias=False
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.dense1(x)
        embed2 = self.gelu(embed1)
        embed2 = self.dense2(embed2)
        embed2 = self.drop(embed2)
        embeds = self.layer_norm(embed1 + embed2)
        return embeds






