from torch.utils.data import Dataset
import typing
import cv2
import typing

class ContrastiveDataset(Dataset):
    """
    Implementation of the dataset,
    for contrastive learning.

    Parameters:
    -----------

        # core parameters

        image_paths: list of input image paths
        title_paths: list of paths to .txt documents, containing titles
        description_paths: list of paths to .txt documents, containing descriptions
        labels - list of corresponding labels
        dataset_type - type of the input dataset, either 'train' or 'valid', depending
        on what is your current goal.

        # augmentations

        image_transformations - input image data augmentations.
        title_transformations - input title sentence augmentations.
        description_transformations - input description sentence augmentations.

        NOTE:
            if you don't want to apply any augmentations to either image or text data.
            you still have to pass
            you still should at least pass following transformations:

                1. for 'image_transformations' arg - normalization and resize.
                2. for 'description_transformations' and 'title_transformations' - their corresponding tokenizers.

                Example:
                    for image data using 'albumentations' library, 

                    see documentation: https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms

                        image_transformations = albumentations.Compose(
                            transforms=[
                                albumentations.Resize(resize_height, resize_width, interpolation),
                                albumentations.Normalize(
                                    mean=means,
                                    std=stds
                                )
                            ]
                        )
                    
                    for description and text data using 'nlaug' library, 

                    see documentation: https://nlpaug.readthedocs.io/en/latest/overview/overview.html

                    description_transformations = naf.Sequential([InputWordTokenizerA])
                    title_transformations = naf.Sequential([InputWordTokenizerB]
    """
    def __init__(self, 
        image_paths: typing.List[str], 
        title_doc_paths: typing.List[str],
        description_doc_paths: typing.List[str],
        labels: typing.List,
        dataset_type: typing.Literal['train', 'valid'],
        image_transformations,
        title_transformations, 
        description_transformations,
        access_document_mode: typing.Literal['r', 'rb']
    ):
        super(ContrastiveDataset, self).__init__()
        self.image_paths = image_paths
        self.title_doc_paths = title_doc_paths 
        self.description_doc_paths = description_doc_paths
        self.labels = labels
        self._dataset_type = dataset_type
        self.image_transformations = image_transformations
        self.title_transformations = title_transformations 
        self.description_transformations = description_transformations
        self.access_file_mode = access_document_mode
    
    def __getitem__(self, idx: int):

        try:
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
            title = open(path=self.title_doc_paths[idx], mode=self.access_file_mode).read()
            description = open(self.description_doc_paths[idx], mode=self.access_file_mode).read()
        
        except(FileNotFoundError) as err:
            raise RuntimeError("failed to load some modalities due to invalid file path.")

        label = self.labels[idx]
        
        try:
            augmented_image = self.image_transformations(image=image)['image'] # augmented image
            tokenized_title = self.title_transformations(title) # augmented and tokenized title embedding
            tokenized_description = self.description_transformations(description) # augmented and tokenized 

        except(Exception) as err:
            logger.error(err)
            raise RuntimeError("failed to apply augmentations to certain modalities. ERROR: '%s'")
        
        if (
                not isinstance(tokenized_title, torch.Tensor) and 
                not isinstance(tokenized_title, numpy.ndarray)
            ):
                raise RuntimeError("""
                failed to tokenize title document, 
                make sure you passed tokenizer 
                to the 'description_transformations'""")

        if (
                not isinstance(tokenized_description, torch.Tensor) and 
                not isinstance(tokenized_description, numpy.ndarray)
            ):
                raise RuntimeError("""
                failed to tokenize description document, 
                make sure you passed tokenizer 
                to the 'description_transformations'""")

        return (
            augmented_image, 
            tokenized_description, 
            tokenized_title, 
            label
        )

    def __len__(self):
        return len(self.image_paths)

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, new_dataset_type: str):
        self._dataset_type = new_dataset_type

class FusionDataset(Dataset):
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


class QuantizationImageDataset(Dataset):
    """
    Dataset for storing image data
    for quantizing image modality encoder.

    NOTE:
        image_paths should match original image data,
        used during training.

    While you can't track batches, there is still some negligible
    error is going to be presented, due to diversity in augmentations.
    """
    def __init__(self, 
        image_paths: typing.List[typing.Union[str, pathlib.Path]], 
        labels: typing.List[typing.Any],
        image_transformations
    ):
        super(QuantizationImageDataset, self).__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.image_transformations = image_transformations
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        label = self.labels[idx]
        augmented_image = self.image_transformations(image=image)['image']
        augmented_image = torch.from_numpy(image).float()
        return augmented_image, label

class QuantizationDescriptionDataset(base.BaseDataset, Dataset):
    """
    Dataset for storing description document text data,
    used for quantizing description modality encoder.
    """
    def __init__(self, 
        description_paths: typing.List[typing.Union[str, pathlib.Path]], 
        labels: typing.List[typing.Any], 
        description_transformations,
        access_file_mode: typing.Literal['r', 'rb'] = 'r',
    ):
        super(QuantizationDescriptionDataset, self).__init__()
        self.description_doc_paths: list = description_paths
        self.labels = labels
        self.description_transformations = description_transformations
        self.access_file_mode: str = access_file_mode

    def __len__(self):
        return len(self.description_paths)

    def __getitem__(self, idx: int):
        try:
            description_text = open(
                self.description_doc_paths[idx], 
                mode=self.access_file_mode).read()

            label = self.labels[idx]
            tokenized_description = self.description_transformations(description_text)

            if (
                not isinstance(tokenized_description, torch.Tensor) and 
                not isinstance(tokenized_description, numpy.ndarray)
            ):
                raise RuntimeError("""
                failed to tokenize description document, 
                make sure you passed tokenizer 
                to the 'description_transformations'""")
            
            return tokenized_description, label

        except(RuntimeError) as err:
            raise err 

        except(FileNotFoundError) as err:
            raise RuntimeError("invalid document paths passed. Document: %s cannot be accessed")

class QuantizationTitleDataset(Dataset):
    """
    Dataset stores title .txt documents, that 
    are used for quantizing title modality encoder.
    """
    def __init__(self, 
        title_doc_paths: typing.List[typing.Union[str, pathlib.Path]], 
        labels: typing.List[typing.Any],
        title_transformations,
        access_file_mode: typing.Literal['r', 'rb'] = 'r'
    ):
        super(QuantizationTitleDataset, self).__init__()
        self.title_doc_paths: list = title_doc_paths
        self.labels: list = labels
        self.title_transformations = title_transformations
        self.access_file_mode = access_file_mode

    def __len__(self):
        return len(self.description_paths)

    def __getitem__(self, idx: int):
        try:
            title_text = open(
                self.title_doc_paths[idx], 
                mode=self.access_file_mode).read()

            label = self.labels[idx]
            tokenized_title = self.title_transformations(title_text)

            if (
                not isinstance(tokenized_title, torch.Tensor) and 
                not isinstance(tokenized_title, numpy.ndarray)
            ):
                raise RuntimeError("""
                failed to tokenize title document, 
                make sure you passed tokenizer 
                to the 'description_transformations'""")

            return tokenized_title, label 

        except(RuntimeError) as err:
            raise err 

        except(FileNotFoundError) as err:
            raise RuntimeError("invalid document paths passed. Document: %s cannot be accessed")

