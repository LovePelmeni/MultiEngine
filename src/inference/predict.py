from src.multimodal import multimodal_net
from src.search.searcher import RecommenderSearchPipeline
import torch
import pathlib
from torch import nn
import typing
import faiss
import albumentations
from src.preprocessing.image_augmentations import ImageIsotropicResize

class InferenceModel(nn.Module):
    """
    Multimodal inference model for 
    predicting similar products, based on the data
    from image and text modalities.
    
    Parameters:
    -----------
        mm_config (typing.Dict) - multimodal encoder configuration.
        search_config (typing.Dict) - similarity search configuration.
    """
    def __init__(self, 
        mm_config: typing.Dict,
        search_config: typing.Dict, 
        preprocess_config: typing.Dict
    ):
        super(InferenceModel, self).__init__()
        # loading image modality preprocessing configuration
        try:
            image_resize_height = preprocess_config.get("resize_height")
            image_resize_width = preprocess_config.get("resize_width")
            image_normalization_means = preprocess_config.get("norm_means")
            image_normalization_stds = preprocess_config.get("norm_stds")
            image_interpolation_up = preprocess_config.get("interpolation_up")
            image_interpolation_down = preprocess_config.get("interpolation_down")

            self.image_augmentations = albumentations.Compose(
                transforms=[
                    ImageIsotropicResize(
                        new_height=image_resize_height, 
                        new_width=image_resize_width,
                        interpolation_up=image_interpolation_up,
                        interpolation_down=image_interpolation_down,
                    ),
                    albumentations.Normalize(
                        mean=image_normalization_means,
                        std=image_normalization_stds
                    )
                ]
            )
        except(KeyError):
            raise RuntimeError("some crucial image preprocessing config parameters are missing")

        # loading text modality preprocessing augmentations
        try:
            pass
        except(KeyError):
            raise RuntimeError("some crucial text preprocessing config parameters are missing")

        # loading multimodal encoder network 
        try:
            image_encoder_path = mm_config.get("image_encoder_path")
            description_encoder_path = mm_config.get("description_encoder_path")
            title_encoder_path = mm_config.get("title_encoder_path")
            fusion_layer_path = mm_config.get("fusion_layer_path")

            self.encoder_net = self.load_multimodal_encoder(
                image_encoder_path=image_encoder_path,
                text_encoder_path=text_encoder_path,
                fusion_layer_path=fusion_layer_path
            )
        except(KeyError):
            raise RuntimeError("some crucial multimodal config parameters are missing")
        
        # loading similarity search index
        try:
            search_index_path = search_config.get("search_index_path")
            refiner_path = search_config.get("refiner_path")
            search_data_path = search_config.get("search_data_path")
            metadata_data_path = search_config.get("metadata_data_path")

            pretrained_search_index = faiss.read_index(search_index_path)
            pretrained_refiner = faiss.read_index(refiner_path)

            self.searcher = self.load_similarity_rec_search(
                pretrained_search_index=pretrained_search_index,
                pretrained_pred_refiner=pretrained_refiner,
                search_dataset_path=search_data_path,
                metadata_search_dataset_path=metadata_data_path
            )
        except(KeyError):
            raise RuntimeError("some crucial similarity search parameters are missing")

    def load_multimodal_encoder(self, 
        image_encoder_path: typing.Union[str, pathlib.Path], 
        title_encoder_path: typing.Union[str, pathlib.Path],
        desc_encoder_path: typing.Union[str, pathlib.Path],
        fusion_layer_path: typing.Union[str, pathlib.Path],
        embedding_length: int
    ):
        """
        Loads multimodal recommendation network, based
        on the provided resource paths.
        
        Parameters:
        -----------
            image_encoder_path - path to the image encoder network 
            text_encoder_path - path to the text encoder network 
            fusion_layer_path - path to the pretrained fusion layer.
            embedding_length - embedding length to use for intermediate
            representation.
        """
        pretrained_image_encoder = torch.load(image_encoder_path)
        pretrained_description_encoder = torch.load(description_encoder_path)
        pretrained_title_encoder = torch.load(title_encoder_path)
        pretrained_fusion_layer = torch.load(fusion_layer_path)

        return multimodal_net.MultimodalNetwork(
            image_encoder=pretrained_image_encoder,
            description_encoder=pretrained_description_encoder,
            title_encoder=pretrained_title_encoder,
            fusion_layer=pretrained_fusion_layer,
            embedding_length=embedding_length,
        )
    
    def load_similarity_rec_search(self, 
        pretrained_search_index: faiss.Index,
        pretrained_pred_refiner: faiss.Index,
        search_dataset_path: typing.Union[str, pathlib.Path],
        metadata_search_dataset_path: typing.Union[str, pathlib.Path],
        init_transform=None,
    ):
        """
        Loads similarity search recommendation 
        algorithm for finding similar product embeddings.
        
        Parameters:
        -----------
            pretrained_search_index - faiss.Index or Composite Index, pretrained
            on a set of product embeddings.
            s
        """
        return RecommenderSearchPipeline(
            search_index=pretrained_search_index,
            search_dataset_path=search_dataset_path,
            metadata_search_dataset_path=metadata_search_dataset_path,
            init_transform=init_transform,
            refiner=pretrained_pred_refiner,
            filtering=None
        )

    def prep_image_data(self, input_images: typing.List[torch.Tensor]):
        """
        Prepares input inference data before passing
        it to the model.
        
        Parameters:
        ----------
            input_images - typing.List - list of input images to apply
        """
        preped_images = []
        for image in input_images:
            prep_image = self.image_augmentations(image)
            prep_image = torch.from_numpy(
            prep_image).permute(2, 0, 1).unsqueeze(0)
            preped_images.append(prep_image)
        return preped_images
            
    def prep_text_data(self, input_texts: typing.List[str]):
        """
        Application string container string validator.
        
        Parameters:
        -----------
            input_texts - list of text descriptions, corresponding
            to image modality.
        """
        preped_texts = []
        for text in input_texts:
            prep_text = self.text_augmentations(text)
            preped_texts.append(prep_text)
        return preped_texts

    def forward(self, 
        input_image: torch.Tensor, 
        input_description: torch.Tensor
    ):
        preped_image = self.prep_image_data(input_images=[input_image])
        preped_text = self.prep_text_data(input_texts=[input_description])

        output_emb: torch.Tensor = self.encoder_net.forward(
            input_image=preped_image,
            input_text=preped_text
        )

        similar_products: typing.List[typing.Dict] = (
            self.searcher.forward(
            input_embedding=output_emb)
        )

        return similar_products

