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
        self.fc = projection.ProjectionLayer(
            in_dim=self.out_bert_dim, 
            out_dim=embedding_length,
            dropout_prob=dropout_prob
        )
        self.feature_extractor.eval()

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
        proj_emb = self.fc(predicted_vector)
        return proj_emb