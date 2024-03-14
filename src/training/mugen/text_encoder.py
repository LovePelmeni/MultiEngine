from torch import nn

class TextEncoder(nn.Module):
    """
    DistilBERT-based Embedding Generation 
    network for processing text sequences
    """
    def __init__(self):
        super(TextEncoder, self).__init__()
