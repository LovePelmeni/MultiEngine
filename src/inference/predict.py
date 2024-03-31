from src.training.mugen import multimodal_net
from training.mugen.image_encoder import ImageEncoder
from src.training.mugen.text_encoder import TextEncoder
from src.training.mugen.audio_encoder import AudioEncoder
from src.training.face_detection import face_detection
from src.training.classifiers import classifiers
from src.training.fusions import fusion_layers
import torch
import json
import pathlib
from glob import glob

class InferenceModel(object):
    """
    Class for running multimodal inference.
    Parameters:
    ----------
    """
    def from_config(cls, json_inference_config_path: pathlib.Path):
         
        config_path = glob(root_dir=json_inference_config_path, recursive=True)
        json_config = json.load(fp=config_path)

        input_img_size = json_config['detector']['input_image_size']
        min_face_size = json_config['detector']['min_face_size']
        face_margin = json_config['detector']['face_margin']
        detector_inference_device = json_config['detector']['inference_device']

        multimodal_network_inference_device = json_config['multimodal']['inference_device']
        embedding_length = json_config['multimodal']['embedding_length']
        output_classes = json_config['multimodal']['output_classes']

        img_input_channels = json_config['multimodal']['image_encoder']['image_input_channels']
        img_input_size = json_config['multimodal']['image_encoder']['image_input_size']

        image_encoder = ImageEncoder(
            input_channels=img_input_channels, 
            input_img_size=img_input_size
        )

        text_encoder = TextEncoder(
            
        )

        audio_encoder = AudioEncoder(

        )

        classifier = classifiers.MultiLayerPerceptronClassifier(
            embedding_length=embedding_length, 
            output_classes=output_classes
        )

        fusion_layer = fusion_layers.AdditiveFusion(latent_size=embedding_length)

        cls.face_detector = face_detection.HumanFaceDetector(
            input_img_size=input_img_size,
            min_face_size=min_face_size,
            face_margin=face_margin,
            inference_device=detector_inference_device,
        )

        cls.model = multimodal_net.MultimodalNetwork(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            audio_encoder=audio_encoder,
            fusion_layer=fusion_layer,
            classifier=classifier,
            embedding_length=embedding_length,
        ).to(multimodal_network_inference_device)

        return cls()

    def predict(self, 
        input_video: torch.Tensor = None, 
        input_audio: torch.Tensor = None,
        input_text: torch.Tensor = None
    ):
        # return dict of probabilities, corresponding 
        # to each individual emotion.
        predicted_labels = self.model.forward(
            input_video=input_video, 
            input_text=input_text,
            input_audio=input_audio
        )
        return predicted_labels

