from urllib.parse import scheme_chars
import faiss
import typing
import torch
import logging
from torch import nn
from src.training.search import search_dataset
import pathlib
import numpy
import pandas

logger = logging.getLogger(name='search_logger')
file_handler = logging.FileHandler(filename="search_logs.log")
logger.addHandler(file_handler)

class RecommenderSearchIndex(nn.Module):
    """
    Module for similarity search of similar
    target embeddings.

    Parameters:
    ----------
        input_dim - input dimension of the embedding
        inv_centroids - number of centroids for Inverted File Index
        top_k - number of similar vectors to search for.
        pq_nbits - number of bits for Product Quantization
        pq_subvecs - number of subvectors for Product Quantization
        top_n_centroids - number of centroids to search from in Inverted File Index.
    """
    def __init__(self,
        input_dim: int, 
        top_k: int,
        inv_centroids: int, 
        pq_nbits: int, 
        pq_subvecs: int,
        top_n_centroids: int, 
    ):
        super(RecommenderSearchIndex, self).__init__()
        assert self.input_dim % pq_subvecs == 0 
        self.input_dim = input_dim
        self.n_subvecs: int = pq_subvecs
        self.top_k: int = top_k
        self.index = faiss.index_factory(
        "IVF%s, PQ%sx%s" % (inv_centroids, pq_subvecs, pq_nbits))
        ivf_index = faiss.extract_index_ivf(self.index)
        ivf_index.nprobe = top_n_centroids

    def forward(self, input_embedding: torch.Tensor):
        if not self.index.is_trained:
            raise ValueError("index is not trained to make predictions.")
        try:
            assert input_embedding.shape[1] % self.pq_subvecs == 0
            _, indices = self.index.search(input_embedding, self.top_k)
            return indices
        except(AssertionError):
            raise RuntimeError("invalid dimensionality of the output vector")

        except(Exception) as err:
            logger.error(err)
            return []


class MetadataPostFiltering(nn.Module):
    """
    Metadata-based post filtering module
    for controlling and maintaining relevance level 
    of the documents after nearest has been found.

    Parameters:
    -----------
        meta_dataset - metadata dataframe.
        rec_dataset - dataset of numpy embedding vectors
    """
    def __init__(self, 
        meta_dataset: pandas.DataFrame, 
        rec_dataset: numpy.ndarray,
    ):
        super(MetadataPostFiltering, self).__init__()
        self.rec_dataset = rec_dataset 
        self.meta_dataset = meta_dataset
    
    def filter_quantitative(self, indices: list, input_prop: str, greater: bool, threshold: float = None):
        if greater:
            return numpy.argwhere(
                self.meta_dataset[
                self.meta_dataset[input_prop].iloc[indices] > threshold
            ])
        else:
            return numpy.argwhere(
                self.meta_dataset[
                self.meta_dataset[input_prop].iloc[indices] > threshold
            ])
    
    def filter_qualitative(self, indices: list, input_prop: str, category: str) -> None:
        new_indices = numpy.argwhere(
            self.meta_dataset[
                self.meta_dataset[input_prop].iloc[indices] == category
            ]
        )
        return new_indices

class RecommenderSearchPipeline(nn.Module):
    """
    Pipeline for searching similar embedding vectors
    in a vector database of products / posts.

    Parameters:
    -----------
        init_transform - (faiss.VectorTransform) preprocessing vector transformation.
        search_index - (faiss.Index) index to use for similarity search.
        refiner (faiss.IndexRefine) another similarity search algorithm to enhance 
        search query of the output of 'search_index'.
    """
    def __init__(self, 
        search_index: faiss.Index,
        search_dataset_path: typing.Union[str, pathlib.Path],
        label_search_dataset_path: typing.Union[str, pathlib.Path],
        init_transform: faiss.VectorTransform = None, 
        refiner: faiss.IndexRefine = None,
        filtering: MetadataPostFiltering = None
    ):
        super(RecommenderSearchPipeline, self).__init__()

        self.index_transform = init_transform
        self.search_dataset = search_dataset.SearchVectorDataset(
            emb_dataset_path=search_dataset_path,
            label_dataset_path=label_search_dataset_path,
            access_mode="r",
            emb_data_shape=None,
            label_data_shape=None
        )
        self.filtering = filtering
        self.search_index = search_index
        self.refiner = refiner
    
    def forward(self, input_embedding: torch.Tensor):

        # apply preprocessing transformation
        processed_embs = self.index_transform(input_embedding)
        # make seach query using similarity index
        searched_query = self.search_index(processed_embs)

        # apply post filtering to put away irrelevant embeddings

        # if self.filtering is not None:
        #     refined_embs_indices = self.filtering(refined_embs_indices)

        # apply refinement search 
        refined_embs_indices = self.refiner(searched_query)

        # parsing data from the vector database file
        output_embs = self.search_dataset._mem_vec_data[refined_embs_indices]
        return output_embs
