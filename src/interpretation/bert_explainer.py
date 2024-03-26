from torch import nn
from captum.attr import IntegratedGradients
import typing
import torch
import matplotlib.pyplot as plt

class TextBertExplainer(object):
    """
    Base module for interpreting text-based
    embedding generation networks.
    """
    def __init__(self, 
        bert_encoder: nn.Module, 
        inference_device: typing.Literal['cuda', 'cpu', 'mps']
    ):
        self.bert_encoder = bert_encoder
        self.inference_device = inference_device
        self.bert_encoder.eval()

    def _visualize_word_attributions(self, predicted_word_weights: typing.Dict) -> None:
        """
        Visualizes importance of each
        provided word as a bar plot.
        
        Parameters:
        -----------
            predicted_word_weights - dictionary-like object
            of format {'word': weight}
        """
        plt.figure(figsize=(15, 10))
        plt.bar(x=list(predicted_word_weights.values()))
        plt.legend(list(predicted_word_weights.keys()))
    
    def explain(self, 
        sentence: str,
        sentence_embedding: torch.Tensor, 
        true_label: int,
        keywords: typing.List[str] = []
    ) -> None:
        """
        Performs explanation qualitative analysis of the 
        model, given input data with keywords to pay attention to.
        NOTE:
            augmented_input_sentence - stands for 
            sentence, which is already augmented
            and ready to be passed to the encoder.
        """
        sentence_words = sentence.strip("!?.,").split(" ")
        gradients = IntegratedGradients(forward_func=self.bert_encoder)
        attributions = gradients.attribute(inputs=sentence_embedding, target=true_label)
        attribution_info = {word: idx for word, idx in zip(sentence_words, attributions)}
        keywords_info = {
            word: weight 
            for word, weight in attribution_info.items() 
            if word.lower() in keywords
        }
        self.visualize_word_importance(keywords_info)

