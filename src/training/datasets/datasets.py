from torch.utils.data import Dataset
from src.training.datasets import base
import typing
import cv2
import typing

class ContrastiveDataset(base.BaseDataset, Dataset):
    """
    Implementation of the dataset,
    for contrastive learning.

    Parameters:
    -----------
        input_paths - list of paths of cropped out human faces
        labels - list of corresponding emotion labels
        transformations (optional) - additional set of image transformations
        to apply to images during training.
    """
    def __init__(self, 
        image_paths: typing.List[str], 
        title_paths: typing.List[str],
        description_paths: typing.List[str],
        labels: typing.List,
        dataset_type: typing.Literal['train', 'valid'],
        image_transformations=None,
        title_transformations=None,
        description_transformations=None
    ):
        super(ContrastiveDataset, self).__init__()
        self.image_paths = image_paths
        self.title_paths = title_paths 
        self.description_paths = description_paths
        self.labels = labels
        self._dataset_type = dataset_type
        self.image_transformations = image_transformations
        self.title_transformations = title_transformations 
        self.description_transformations = description_transformations
    
    def __getitem__(self, idx: int):

        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        title = open(path=self.audio_paths[idx], mode='r').read()
        description = open(self.description_paths[idx], mode='r').read()
        
        label = self._input_labels[idx]

        if self.video_transformations is not None:
            image = self.image_transformations(image=image)['image']

        if self.text_transformations is not None:
            title = self.title_transformations(title)

        if self.audio_transformations is not None:
            description = self.description_transformations(description)

        return image, description, title, label

    def __len__(self):
        return len(self.image_paths)

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, new_dataset_type: str):
        self._dataset_type = new_dataset_type


class FusionDataset(data.Dataset):
    """
    Implementation of the dataset 
    for training fusion layer of multimodal
    network.

    Parameters:
    -----------
        image_embedding_path - path to the .dat or .bin file where image modality embeddings are stored
        desc_emebedding_path - path to the .dat or .bin file where description modality embeddings are stored
        title_embedding_path - path to the .dat or .bin file where title modality embeddings are stored
        image_emb_type - dtype of image embeddings
        title_emb_type - dtype of title embeddings
        desc_emb_type - dtype of description embeddings
        labels - list of corresponding class labels
    """
    def __init__(self, 
        image_embedding_path: pathlib.Path, 
        desc_embedding_path: pathlib.Path, 
        title_embedding_path: pathlib.Path,
        labels: typing.List,
        image_emb_type: numpy.dtype,
        desc_emb_type: numpy.dtype,
        title_emb_type: numpy.dtype
    ):
        super(FusionDataset, self).__init__()

        self.image_embeddings = numpy.memmap(
            image_embedding_path, 
            dtype=img_emb_type
        )
        self.desc_embeddings = numpy.memmap(
            desc_embedding_path, 
            dtype=desc_emb_type
        )
        self.title_embeddings = numpy.memmap(
            title_embedding_path, 
            dtype=title_emb_type
        )
        self.labels: typing.List = labels
    
    def __getitem__(self, idx: int):
        img_emb = self.image_embeddings[idx]
        title_emb = self.title_embeddings[idx]
        desc_emb = self.desc_embeddings[idx]
        label = self.labels[idx]
        return img_emb, desc_emb, title_emb, label

    def __len__(self):
        return len(self.image_embedding)



