from torch import nn
import typing
from transformers.models import bert
from src.multimodal import projection

class TextEncoder(nn.Module):
    """
    DistilBERT-based Embedding Generation 
    network for processing text sequences
    """
    def __init__(self, 
        bert_model: bert.BertModel, 
        embedding_length: int,
        bert_tokenizer: bert.BertTokenizer,
        dropout_prob: float = 0.1
    ):
        super(TextEncoder, self).__init__()
        self.feature_extractor = bert_model
        self.tokenizer = bert_tokenizer
        self.out_bert_dim = self.feature_extractor.config.hidden_size
        self.proj_head = projection.ProjectionLayer(
            in_dim=self.out_bert_dim, 
            out_dim=embedding_length,
            dropout_prob=dropout_prob
        )
        self.tokenizer.eval()
        self.feature_extractor.eval()
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
        input_sentences: typing.List[str], 
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None, 
        head_mask=None
    ):
        tokenized_input = self.tokenizer(text=input_sentences)
        predicted_vector = self.feature_extractor(
            tokenized_input,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        proj_emb = self.proj_head(predicted_vector)
        return proj_emb




