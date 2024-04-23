from torch import nn

class MultiplicativeFusion(nn.Module):
    """
    Implementation of the basic multiplicative
    fusion layer for merging features from multiple 
    modalities.
    """
    def __init__(self):
        super(MultiplicativeFusion, self).__init__()
        
    def forward(self, embeddings: typing.List[torch.Tensor]):
        output_emb = embeddings[0]
        for emb in range(1, len(embeddings)):
            output_emb = torch.multiply(output_emb, embeddings[emb])
        return output_emb