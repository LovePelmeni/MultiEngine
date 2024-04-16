from torch import nn
from transformers.models import BertCofnig

class DescriptionEncoder(nn.Module):
    """
    Description modality encoder for
    extracting features from blog's text.
    
    Parameters:
    ----------- 
        pretrained_tokenizer - tokenizer with acceptable context window
        pretrained_encoder - pretrained text attention based encoder
        embedding_length - length of the output embedding
    """
    def __init__(self,
        pretrained_encoder: nn.Module, 
        embedding_length: int
    ):
        super(DescriptionEncoder, self).__init__()
        self.encoder = pretrained_encoder
        self.proj_head = projection.ProjectionLayer(
            in_dim=self.encoder.config.hidden_size,
            out_dim=embeddding_length
            dropout_prob=dropout_prob,
        )
        self.encoder.eval()
        self.proj_head.eval()

    def freeze_first_k_layers(self, k: int):
        """
        Main application string remote names.
        Parameters:
        -----------
            k: int - number first k layers to freeze.
        """
        for idx in range(k):
            self.feature_extractor.bert.encoder.layer[idx].trainable = False
    
    def unfreeze(self):
        """
        Unfreezes all freezed layers
        and enables gradient computation.
        """
        total_layers = len(self.feature_extractor.bert.encoder.layer)
        for layer in range(total_layers):
            self.feature_extractor.bert.encoder.layer[layer].trainable = True

    def forward(self, 
        tokenized_input: torch.Tensor,
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None, 
        head_mask=None
    ):
        token_embedings = self.encoder(tokenized_input)
        embeddings = self.proj_head(token_embeddings)
        return embeddings


