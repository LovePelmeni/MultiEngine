from torch import nn
import typing
import torch 

class MultiplicativeFusion(nn.Module):
    """
    Layer for fusion of multimodal embeddings
    using multiplicative fusioning approach
    """
    def __init__(self, latent_size: int):
        super(MultiplicativeFusion, self).__init__()
        self.latent_size = latent_size

    def forward(self, embeddings: typing.List[torch.Tensor]):

        if embeddings.shape[0] == 0:
            return None

        output_emb = torch.empty(embeddings[0].shape)
        for emb in embeddings:
            output_emb = torch.multiply(output_emb, emb)
        return output_emb

class AdditiveFusion(nn.Module):
    """
    Layer for fusion of multimodal embeddings
    using additive fusioning approach
    """
    def __init__(self, latent_size: int):
        super(AdditiveFusion, self).__init__()   
        self.latent_size = latent_size

    def forward(self, embeddings: typing.List[torch.Tensor]):

        if embeddings[0].shape[0] == 0:
            return None

        output_emb = torch.empty(embeddings[0].shape)
        for emb in embeddings:
            output_emb = torch.add(output_emb, emb)
        return output_emb