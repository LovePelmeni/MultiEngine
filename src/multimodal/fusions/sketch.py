from torch import nn
import logging 


class MultimodalSketching(nn.Module):
    """
    Implementation of the multimodal 
    sketching strategy for effectively
    merging modalities for feature extraction
    purposes.
    """
    def __init__(self, **kwargs):
        super(MultimodalSketching, self).__init__()

    def forward(self, input_modalities: typing.List[torch.Tensor]):
        pass
