import faiss
import typing
import torch

class SimilarItemSearcher(object):
    """
    Base module for searching
    similar embeddings 
    """
    @classmethod
    def from_config(cls, config: typing.Dict):

        number_of_suggestions = config.get("number_of_suggestions")
        index_paths = config.get("index_paths")

        cls.number_of_suggestions = number_of_suggestions
        cls.indexes: typing.Dict[str, faiss.Index] = {
            index_label: faiss.read_index(index_path)
            for index_label, index_path in index_paths.items()
        }
        return cls()
        
    def search(self, embedding: torch.Tensor, category: str) -> typing.List[int]:
        _, candidate_indices = self.indexes[category].search(
            x=embedding,
            k=self.number_of_suggestions
        )
        return candidate_indices
        