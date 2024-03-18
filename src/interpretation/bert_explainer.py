from src.interpretation import base
from torch import nn
import typing
import numpy
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import (
    PCA,
    KernelPCA
)
import torch
import logging

text_logger = logging.getLogger("text_logger")
handler = logging.FileHandler(filename="text_interpretaton_logs.log")
text_logger.addHandler(handler)

class BertExplainer(base.BaseExplainer):
    """
    Base module for interpreting text 
    processing networks, such as Bert, DistilBERT, etc..
    And provide insights into which context words
    encforces them to generate a specific embedding.

    In case of embedding generation, we simply 
    want to make sure, that corresponding opposite
    sentences have larger distance in vector space,
    while similar ones have smaller.
    """
    def __init__(self, 
        text_model: nn.Module, 
        inference_device: typing.Literal['cpu', 'cuda', 'mps'],
        unique_label_values: typing.List[typing.Union[str, int]],
        word2vec_setup: typing.Dict,
    ):
        self.text_encoder = text_model.to(inference_device)
        self.word2vec_setup = word2vec_setup
        self.inference_device = inference_device
        self.color_map: typing.Dict = {
            label: tuple(numpy.random.randint(low=0, high=255, size=3))
            for label in unique_label_values
        }

    def generate_sentence_input_emb(self, sentences: typing.List[str]):
        model = Word2Vec(
            sentences=sentences, 
            **self.word2vec_setup
        )
        output_embs = []
        for sentence in sentences:
            words = sentence.split()
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            avg_vector = numpy.mean(word_vectors, axis=0)
            output_embs.append(avg_vector)
        return output_embs
    
    @staticmethod
    def generate_sentence_output_emb(self, sentence_emb: torch.Tensor):
        """
        Generates multimodal embedding vector
        from provided text embeddings.
        """
        pred_emb = self.text_encoder.forward(
            sentence_emb.to(self.inference_device).float()
        ).cpu()
        return pred_emb

    def reduce_dims(self, embedding_vectors: numpy.ndarray):
        pca = KernelPCA(n_components=2, kernel="rbf")
        reduced_dim_vecs = pca.fit_transform(embedding_vectors)
        return reduced_dim_vecs

    def visualize_embedding_clusters(self, 
        labels: typing.List,
        gen_embeddings: typing.List[numpy.ndarray],
    ):
        stacked_output = numpy.stack(gen_embeddings, axis=0)
        output_2d_embs = self.reduce_dims(stacked_output)
        unique_labels = numpy.unique(labels)
        for label in unique_labels:
            label_embs = output_2d_embs[numpy.where(labels == label).indices]
            plt.scatter(
                x=label_embs[:, 0], 
                y=label_embs[:, 1], 
                cmap=self.color_map[label]
            )
        plt.legend([label for label in unique_labels])
    
    def forward(self, 
        text_sentences_input: typing.List[str], 
        sentence_labels: typing.List[typing.Union[str, int]]
    ):
        try:
            input_embs = self.generate_sentence_input_emb(text_sentences_input)
            predicted_embeddings = self.text_encoder(input_embs).numpy()
            self.visualize_embedding_clusters(
                labels=sentence_labels, 
                gen_embeddings=predicted_embeddings
            )
        except(Exception) as err:
            text_logger.error(err)