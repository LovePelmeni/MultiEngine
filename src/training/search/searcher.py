import faiss
import typing
import abc
import torch
import numpy
import logging
import pathlib
from src.training.search import search_dataset

logger = logging.getLogger(name='search_logger')
file_handler = logging.FileHandler(filename="search_logs.log")
logger.addHandler(file_handler)


class BaseVectorQuantizer(abc.ABC):
    """
    Base module for quantizing
    embedding vectors to a lower
    representation state
    """
    @abc.abstractproperty
    def _quantizer(self) -> faiss.Quantizer:
        raise NotImplementedError()
 
    @abc.abstractclassmethod
    def from_config(cls, config: typing.Dict):
        """
        Loads vector quantizer from specified
        configuration.

        Parameters:
        -----------
            config - (dict) - dictionary, containing 
            parameters.
        """
    @abc.abstractmethod
    def save(self, index_path: typing.Union[str, pathlib.Path]):
        """
        Saves quantizer under 'name.index'
        application.
        
        Parameters:
        -----------
            index_path - destination path to save index.
        """

    @abc.abstractmethod
    def shrink(self, embedding: numpy.ndarray):
        """
        Shrinks embedding vector to a lower
        dimensional representation.

        Parameters:
        -----------
            embedding - embedding to shrink (must match in dim)
        """


class BaseSimilaritySearcher(abc.ABC):
    """
    Base module for searching similar
    embedding vectors, that are spatial
    representations of products or items to recommend.
    """
    @abc.abstractproperty 
    def _index(self) -> faiss.Index:
        raise NotImplementedError()

    @abc.abstractclassmethod
    def from_config(cls, config: typing.Dict):
        """
        Loads pretrained similarity search index
        instance, based on the specified configuration.
        
        Parameters:
        -----------
            config - typing.Dict object, which contains
            parameters of the similarity search module.
        """

    @abc.abstractmethod
    def train(self, train_embeddings: numpy.ndarray):
        """
        In case Search Index is trainable,
        we should provide method to train
        it on a given set of 'embedding' vectors
        """
        if not self.index.is_trained:
            if hasattr(self.index, 'add'):
                self.index.add(train_embeddings)

    @abc.abstractmethod
    def search(self, embedding: torch.Tensor):
        """
        Performs index search to
        get top K similar embeddings.
        
        Parameters:
        ----------
            embedding - target embedding
            item_info - additional information about item.
        """

class FlatSearcher(BaseSimilaritySearcher):
    """
    Base module for searching similar 
    embedding vectors using L2 euclidian distance norm
    """
    def __init__(self):
        super(BaseSimilaritySearcher, self).__init__()

    @classmethod
    def from_config(cls, config: typing.Dict):

        number_of_suggestions = config.get("number_of_suggestions")

        if 'index_path' in config:
            index_path = config.get("index_path")
            cls._index  = faiss.read_index(index_path)
        else:
            dim = config.get("embedding_dim")
            cls._index = faiss.IndexFlatL2(dim)

        cls._number_of_suggestions = number_of_suggestions

        # loading train dataset
        dataset_path = config.get("dataset_path")
        access_mode = config.get("access_mode")
        data_shape = config.get("data_shape")
        data_type = config.get("data_type")

        cls._dataset = search_dataset.SearchVectorDataset(
            dataset_path=dataset_path,
            access_mode=access_mode,
            data_shape=data_shape,
            data_type=data_type
        )

        if not cls._index.is_trained:
            cls._index.train(cls._dataset._mem_vec_data)
            cls._index.add(cls._dataset.mem_vec_data)
        return cls()
        
    def search(self, embedding: torch.Tensor) -> typing.List[int]:
        _, candidate_indices = self._index.search(
            embedding, 
            self._number_of_suggestions
        )
        return candidate_indices
        
class LSHSearcher(BaseSimilaritySearcher):
    """
    Module for finding similar embedding vectors
    based on Local Similarity Hashing (LSH) algorithm.
    """
    def __init__(self):
        super(BaseSimilaritySearcher, self).__init__()
    
    @classmethod
    def from_config(cls, config: typing.Dict):

        index_path = config.get("index_path")
        num_of_suggestions = config.get("num_suggestions")

        if 'index_path' in config:
            cls._index = faiss.read_index(index_path)
        else:
            num_hash_tables = config.get("num_hash_tables")
            embedding_dim = config.get("embedding_dim")
            cls._index = faiss.IndexLSH(
                embedding_dim, 
                num_hash_tables
            )

        cls._number_of_suggestions = num_of_suggestions

        # loading train dataset
        dataset_path = config.get("dataset_path")
        access_mode = config.get("access_mode")
        data_shape = config.get("data_shape")
        data_type = config.get("data_type")

        cls._dataset = search_dataset.SearchVectorDataset(
            dataset_path=dataset_path,
            access_mode=access_mode,
            data_shape=data_shape,
            data_type=data_type
        )

        if not cls._index.is_trained:
            cls._index.train(cls._dataset._mem_vec_data)
            cls._index.add(cls._dataset.mem_vec_data)
        return cls()

    def search(self, embedding: torch.Tensor) -> typing.List:
        _, candidate_indices = self._index.search(
            embedding, 
            self._number_of_suggestions
        )
        return candidate_indices

class HNSWSearcher(BaseSimilaritySearcher):
    """
    Base module for finding similarity
    between embedding vectors, based on (HNSW)
    searching algorithm.
    """
    def __init__(self):
        super(BaseSimilaritySearcher, self).__init__()

    @classmethod
    def from_config(cls, config: typing.Dict):

        index_path = config.get("index_path")
        num_of_suggestions = config.get("num_suggestions")
        dim = config.get("embedding_dim")

        if 'index_path' in config:
            cls._index = faiss.read_index(index_path)
        else:
            # Set HNSW index parameters
            M = config.get("vertex_connections") # number of connections each vertex will have
            ef_search = config.get("ef_search") # depth of layers explored during search
            ef_construction = config.get("ef_construction") # depth of layers explored during index construction

            cls._index = faiss.IndexHNSW(dim, M)
            cls._index.hnsw.efContruction = ef_construction
            cls._index.hnsw.efSearch = ef_search

        cls._number_of_suggestions = num_of_suggestions
        # loading train dataset
        dataset_path = config.get("dataset_path")
        access_mode = config.get("access_mode")
        data_shape = config.get("data_shape")
        data_type = config.get("data_type")

        cls._dataset = search_dataset.SearchVectorDataset(
            dataset_path=dataset_path,
            access_mode=access_mode,
            data_shape=data_shape,
            data_type=data_type
        )

        if not cls._index.is_trained:
            cls._index.train(cls._dataset._mem_vec_data)
            cls._index.add(cls._dataset._mem_vec_data)
        return cls()

    def search(self, embedding: torch.Tensor) -> typing.List:
        _, candidate_indices = self._index.search(
            embedding, 
            self._number_of_suggestions
        )
        return candidate_indices

class IVNFPQQuantizer(BaseSimilaritySearcher):
    """
    Base module for finding similar embedding vectors
    using Inverted File Product Quantization Algorithm.
    """
    @classmethod
    def from_config(cls, config: typing.Dict):

        num_of_suggestions = config.get("num_suggestions")

        if 'index_path' in config:
            index_path = config.get("index_path")
            cls._quantizer = faiss.read_index(index_path)
        else:
            n_centroids = config.get("n_centroids")
            code_size = config.get("code_size")
            nbits = config.get("nbits")
            embedding_dim = config.get('embedding_dim')

            coarse_quantizer = faiss.IndexFlatL2(embedding_dim)
            cls._quantizer = faiss.IndexIVFPQ(
                coarse_quantizer, 
                embedding_dim, 
                n_centroids, 
                code_size, nbits
            )
        
        # loading train dataset
        dataset_path = config.get("dataset_path")
        access_mode = config.get("access_mode")
        data_shape = config.get("data_shape")
        data_type = config.get("data_type")

        cls._dataset = search_dataset.SearchVectorDataset(
            dataset_path=dataset_path,
            access_mode=access_mode,
            data_shape=data_shape,
            data_type=data_type
        )

        if not cls.quantizer.is_trained:
            cls._quantizer.train(cls._dataset._mem_vec_data)
            cls._quantizer.add(cls._dataset._mem_vec_data)

        cls._number_of_suggestions = num_of_suggestions
        return cls()

    def shrink(self, embedding: torch.Tensor):
        _, candidate_indices = self._quantizer(
            n=self._number_of_suggestions, 
            x=embedding
        )
        return candidate_indices



