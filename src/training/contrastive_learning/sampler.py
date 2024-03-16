from tkinter import E
from torch.utils.data import Sampler
from src.training.contrastive_learning import similarity
import typing
import numpy
import torch
from abc import ABC, abstractmethod

class BaseSampler(Sampler, ABC):
    """
    Base Sampling class for batch training
    of neural networks.
    
    Parameters:
    ----------
        batch_size - size of the batch
        video_data - list of video files
        textual_data - list textual sentence blocks
        audio_data - list of audio files
    """
    def __init__(self, 
        batch_size: int, 
        video_data: typing.List[typing.Union[numpy.ndarray, torch.Tensor]], 
        textual_data: typing.List[typing.Union[numpy.ndarray, torch.Tensor]], 
        audio_data: typing.List[typing.Union[numpy.ndarray, torch.Tensor]],
        labels: typing.List[int],
        **kwargs
    ):
        super(BaseSampler, self).__init__(**kwargs)
        self.batch_size = batch_size 
        self.aligned_pairs, self.labels = self.align_modal_data(
            video_data=video_data,
            text_data=textual_data,
            audio_data=audio_data,
            labels=labels
        )
    
    def __len__(self):
        return len(self.video_data)

    @abstractmethod
    def pair_similarity_metric(self, pair1, pair2, **kwargs):
        """
        Method, which serves as a metric
        for finding similar pairs of multimodal data,
        among batch.
        Warning:
            this is just empty shell, which is implemented
            in other classes
        """

    @abstractmethod
    def hard_mining(self, 
        batch_data: typing.List[
                typing.Tuple[
                    torch.Tensor,
                    torch.Tensor, 
                    torch.Tensor
                ]
            ],
        batch_labels: typing.List,
        **kwargs
    ):
        """
        Method which implements logic 
        of picking hard negative and positive samples
        Warning:
            this is just empty shell, which is implemented
            in other classes.
        """

    def __iter__(self):
        total_batches = len(self.video_data[0]) / self.batch_size
        output_batches = []

        for idx in range(total_batches):
            start = idx*self.batch_size
            end = start+self.batch_size
            batch_data = self.aligned_pairs[start:end]
            batch_labels = self.labels[start:end]
            mined_pairs = self.hard_mining(batch_data=batch_data, batch_labels=batch_labels)
            output_batches.append(mined_pairs)
            
        if len(self.video_data[0]) % self.batch_size != 0:
            output_batches.append(self.aligned_pairs[:end])
        
        return output_batches


class InstanceDiscriminationContrastSampler(BaseSampler):
    
    def __init__(self, 
        video_transformations,
        text_transformations,
        audio_transformations,
        **kwargs
    ):
        super(InstanceDiscriminationContrastSampler, self).__init__(**kwargs)
        self.video_transformations = video_transformations
        self.text_transformations = text_transformations 
        self.audio_transformations = audio_transformations
        
    def hard_mining(self, batch_data: typing.List, batch_labels: typing.List):

        output_samples = []

        for idx, sample in enumerate(batch_data):

            pos_sample = (
                self.video_transformations(sample[0]) if self.video_transformations else sample[0],
                self.text_transformations(sample[1]) if self.text_transformations else sample[1],
                self.audio_transformations(sample[2]) if self.audio_transformations else sample[2]
            )

            neg_sample = sorted([
                pair for pair in range(len(batch_labels))
                if pair != idx and batch_labels[idx] != batch_labels[pair]],
                key=lambda sample_idx: self.pair_similarity_metric(sample, batch_data[sample_idx])
            )[:1]

            output_samples.append(
                (
                    pos_sample,
                    sample,
                    neg_sample
                )
            )
        return output_samples

class SupervisedContrastSampler(BaseSampler):
    """
    Implementation of batch sampler,
    which leverages concept of Supervised Contrastive Learning
    to form batches on the fly
    """
    def __init__(self, **kwargs):
        super(SupervisedContrastSampler, self).__init__(**kwargs)

    def pair_similarity_metric(self, pair1, pair2):
        video_sim = similarity.SSIM().compute(img1=pair1[0], img2=pair2[0])
        text_sim = similarity.calculate_text_similarity(pair1[1], pair2[1])
        audio_sim = similarity.calculate_audio_similarity(pair1[-1], pair2[-1])
        return video_sim + text_sim + audio_sim
        
    def hard_mining(self, batch_data: typing.List, batch_labels: typing.List):
        output_samples = []

        for idx, sample in enumerate(batch_data):

            pos_sample = sorted([
                pair for pair in range(len(batch_data)) 
                if pair != idx and batch_labels[idx] == batch_labels[pair]
            ],
                key=lambda sample_idx: self.pair_similarity_metric(sample, batch_data[sample_idx])
            )[:1]

            neg_sample = sorted([
                pair for pair in range(len(batch_data)) 
                if pair != idx and batch_labels[idx] != batch_labels[pair]
            ],
                key=lambda sample_idx: self.pair_similarity_metric(sample, batch_data[sample_idx])
            )[:1]

            output_samples.append(
                (
                    pos_sample,
                    sample,
                    neg_sample
                )
            )
        return output_samples 



