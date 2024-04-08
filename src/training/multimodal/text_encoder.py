from torch import nn
import typing
from transformers.models import bert

class TextEncoder(nn.Module):
    """
    DistilBERT-based Embedding Generation 
    network for processing text sequences
    """
    def __init__(self, 
        bert_model: bert.BertModel, 
        bert_tokenizer: bert.BertTokenizer
    ):
        super(TextEncoder, self).__init__()
        self.feature_extractor = bert_model
        self.tokenizer = bert_tokenizer
        self.out_dim = self.feature_extractor.config.hidden_size

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
        return predicted_vector