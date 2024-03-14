from torch.utils.data import Dataset
from src.training.datasets import base
from src.preprocessing.text_augmentations import BertTextTransform
from src.preprocessing.video_augmentations import VideoTransform
from torchmultimodal.transforms import (
    text_transforms,
    video_transform,
)

import typing
import cv2
import torch

class VideoEncoderDataset(base.BaseDataset, Dataset):
    """
    Implementation of the dataset,
    compatible with training / evaluating 
    VideoCLIP-based Embedding Generation Network.
    
    Parameters:
    -----------
        input_paths - list of paths of cropped out human faces
        labels - list of corresponding emotion labels
        transformations (optional) - additional set of image transformations
        to apply to images during training.
    """
    def __init__(self, 
        input_paths: typing.List[str], 
        labels: typing.List,
        dataset_type: typing.Literal['train', 'valid'],
        normalization_mean: float,
        normalization_std: float
    ):
        self._input_paths = input_paths
        self._input_labels = labels
        self._dataset_type = dataset_type
        self._transformations = video_transform.VideoTransform(
            mean=normalization_mean,
            std=normalization_std,
        )
    
    def __getitem__(self, idx: int):
        image = cv2.imread(self._input_paths[idx], cv2.IMREAD_UNCHANGED)
        label = self._input_labels[idx]
        if self._transformations is not None:
            image = self._transformations(image)['image']
        image = torch.from_numpy(image).float()
        return image, label

    def __len__(self):
        return len(self._input_paths)

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, new_dataset_type: str):
        self._dataset_type = new_dataset_type

class TextEncoderDataset(base.BaseDataset):
    """
    Textual dataset for storing textual
    data for distilBERT-based embedding generation encoder.
    """
    def __init__(self, ):
        self.transformations = text_transforms.Bert