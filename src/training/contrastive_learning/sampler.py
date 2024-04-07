from src.training.contrastive_learning import similarity
import typing
import numpy
import torch
from abc import ABC, abstractmethod

class BaseSampler(ABC):
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
    ) -> typing.List[typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Method which implements logic 
        of picking hard negative and positive samples
        Warning:
            this is just empty shell, which is implemented
            in other classes.
        """

class InstanceDiscriminationContrastSampler(BaseSampler):
    
    def __init__(self, 
        image_transformations,
        text_transformations,
        audio_transformations,
        **kwargs
    ):
        super(InstanceDiscriminationContrastSampler, self).__init__(**kwargs)
        self.image_transformations = image_transformations
        self.text_transformations = text_transformations 
        self.audio_transformations = audio_transformations
        
    def hard_mining(self, batch_data: typing.List, batch_labels: typing.List):
 
        output_samples = []

        for idx, sample in enumerate(batch_data):

            pos_sample = (
                self.image_transformations(sample[0]) if self.video_transformations else sample[0],
                self.text_transformations(sample[1]) if self.text_transformations else sample[1]
            )

            neg_sample = sorted([
                pair for pair in range(len(batch_labels))
                if pair != idx and batch_labels[idx] != batch_labels[pair]],
                key=lambda sample_idx: self.pair_similarity_metric(sample, batch_data[sample_idx])
            )[0]

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

    def pair_similarity_metric(self, 
        pair1: torch.Tensor, # either image or text sentence
        pair2: torch.Tensor, # either image or text sentence
        data_type: typing.Literal['image', 'text']
    ):
        if data_type == 'image':
            ssim = similarity.SSIM().compute(img1=pair1, img2=pair2)
        elif data_type == 'text':
            ssim = similarity.calculate_text_similarity(pair1, pair2)
        return ssim
        
    def hard_mining(self, batch_data: typing.List, batch_labels: typing.List):
        """
        Finds hard mining samples to enhance training
        of the classifier.
        
        Parameters:
        ----------
            batch_data - data, which belongs to a single modality
            batch_labels - corresponding list of labels.
        """
        output_samples = []
 
        for idx, sample in enumerate(batch_data):

            pos_sample = sorted([
                pair for pair in range(len(batch_data)) 
                if pair != idx and batch_labels[idx] == batch_labels[pair]
            ],
                key=lambda sample_idx: self.pair_similarity_metric(sample, batch_data[sample_idx])
            )[0]

            neg_sample = sorted([
                pair for pair in range(len(batch_data)) 
                if pair != idx and batch_labels[idx] != batch_labels[pair]
            ],
                key=lambda sample_idx: self.pair_similarity_metric(sample, batch_data[sample_idx])
            )[0]

            output_samples.append(
                (
                    pos_sample,
                    sample,
                    neg_sample
                )
            )
        return output_samples

