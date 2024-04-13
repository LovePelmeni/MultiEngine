from sklearn.decomposition import PCA
from src.training.search import base
import typing

class PreprocessingPipeline(object):
    """
    Implementation of the preprocessing
    pipeline for enhancing appearance of the embedding
    vectors.
    
    Parameters:
    ----------- 
        preprocessings - list of preprocessing augmentations,
        that will be applied to vector before searching
        for neighours.
    """
    def __init__(self, preprocessings: typing.List[BaseLinearTransform]):
        self.preprocessings: list = preprocessings

    def apply(self, input_embeddings: typing.Union[torch.Tensor, numpy.ndarray]):
        output = input_embeddings
        for prep in self.preprocessings:
            output = prep(input_embeddings)
        return output

class PreprocessLinearTransform(BaseLinearTransform):
    """
    Implementation of the PCA Linear
    preprocessing for compressing embedding vectors

    Parameters:
    -----------
        n_components - (int) - components for PCA (principal component analysis).
    """
    def __init__(self, n_components: int):
        self.pca = PCA(n_components)
    
    def preprocess(self, input_embeddings: typing.List[
        typing.Union[torch.Tensor, numpy.ndarray]]):
        
        if isinstance(input_embedding, torch.Tensor):
            input_embedding = torch.numpy(input_embeddings)
        transformed_embs = self.pca.fit_transform(input_embeddings)
        return transformed_embs