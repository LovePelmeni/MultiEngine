from torch.utils.data import Dataset
from src.training.datasets import base
from gensim.models import Word2Vec
from librosa import load as audio_load
import typing
import cv2

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
        video_paths: typing.List[str], 
        text_doc_paths: typing.List[str],
        audio_paths: typing.List[str],
        labels: typing.List,
        dataset_type: typing.Literal['train', 'valid'],
        video_transformations=None,
        text_transformations=None,
        audio_transformations=None
    ):
        super(ContrastiveDataset, self).__init__()
        self.video_paths = video_paths
        self.text_doc_paths = text_doc_paths 
        self.audio_paths = audio_paths
        self.labels = labels
        self._dataset_type = dataset_type
        self.video_transformations = video_transformations
        self.text_transformations = text_transformations 
        self.audio_transformations = audio_transformations
    
    def __getitem__(self, idx: int):

        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        text = Word2Vec(sentences=self.text_doc_paths[idx],)
        audio = audio_load(path=self.audio_paths[idx], offset=0)
        label = self._input_labels[idx]

        if self.video_transformations is not None:
            video = self.video_transformations(img)['video']

        if self.text_transformations is not None:
            text = self.text_transformations(text)

        if self.audio_transformations is not None:
            audio = self.audio_transformations(audio)

        return video, text, audio, label

    def __len__(self):
        return len(self._input_paths)

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, new_dataset_type: str):
        self._dataset_type = new_dataset_type


