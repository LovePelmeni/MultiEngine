from torch import nn
from src.training.mugen import projection
from src.training.search import searcher
from src.query.embeddings import SimilarEmbeddingRetriever
from src.training.classifiers import classifiers
import torch
import typing

class MultimodalNetwork(nn.Module):
    """
    Multimodal network for handling 
    data from multiple sources (modalities)
    including video, text and audio emotion
    information.

    Parameters:
    -----------
        video_encoder - network for processing video and generating video embeddings
        text_encoder - network for processing text units and generating text embeddings
        audio_encoder - network for processing audio sequences and generating audio embeddings
        fusion_layer - layer for fusing embeddings and aligning them accordingly.
    """
    def __init__(self, 
        image_encoder: nn.Module, 
        text_encoder: nn.Module, 
        fusion_layer: nn.Module,
        embedding_length: int,
        output_classes: int,
        faiss_config: typing.Dict
    ):
        super(MultimodalNetwork, self).__init__()

        self.image_encoder = nn.Sequential(
            image_encoder,
            projection.ProjectionLayer(
                in_dim=image_encoder.out_dim, 
                out_dim=embedding_length
            )
        )
        self.text_encoder = nn.Sequential(
            text_encoder,
            projection.ProjectionLayer(
                in_dim=text_encoder.out_dim,
                out_dim=embedding_length
            )
        )
        self.fusion_layer = fusion_layer
        self.classifier = classifiers.MultiLayerPerceptronClassifier(
            embedding_length=embedding_length,
            output_classes=output_classes,
        )
        self.embedding_retriever = SimilarEmbeddingRetriever()
        self.faiss_searcher = searcher.SimilarItemSearcher.from_config(config=faiss_config)

    def forward(self, 
        input_video: torch.Tensor = None, 
        input_text: torch.Tensor = None,
    ) -> torch.Tensor:
        fused_embs = self.fusion_layer(
            modalities=[input_video, input_text],
            classifiers=[self.video_encoder, self.text_encoder]
        )
        # embeddings unfiltered population, similar to fused_emb by information attributes
        category_embs = self.embedding_retriever.retrieve_similar_embs(fused_embs)

        # first K most similar embeddings found in population
        k_similar_embs_indices = self.faiss_searcher.search(
            embedding=fused_embs, 
            embedding_pop=category_embs
        )
        # filtered embeddings information after applying FAISS algorithm
        filtered_embeddings = [
            category_embs[emb_idx] for emb_idx in range(len(category_embs))
            if emb_idx in k_similar_embs_indices
        ]
        return filtered_embeddings