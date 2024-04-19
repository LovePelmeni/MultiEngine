from src.preprocessing import base
from torchvision.transforms.functional import normalize, resize
import torch
import typing
import cv2
import albumentations
import numpy

"""
NOTE:   
    mean and std, selected for normalization
    may vary, depending on image modality encoder you use.
    you may need to override DEFAULT_MEAN and DEFAULT_STD
    environment variables to specify which parameters to use.
"""
DEFAULT_MEAN = os.environ.get("DEFAULT_NORM_MEAN", (0.43216, 0.394666, 0.37645))
DEFAULT_STD = os.environ.get("DEFAULT_NORM_STD", (0.22803, 0.22145, 0.216989))

class ImageIsotropicResize(albumentations.ImageOnlyTransform):
    """
    Base module for resizing
    image in a 
    """
    def __init__(self, 
        target_size: int,
        interpolation_up = cv2.INTER_LINEAR, 
        interpolation_down = cv2.INTER_CUBIC
    ):
        self.target_size = target_size
        self.interpolation_up = interpolation_up
        self.interpolation_down = interpolation_down

    def apply(self, input_img: numpy.ndarray):

        height, width = input_img.shape[:2]
        if (self.target_size, self.target_size) == (height, width):
            return input_img

        if height > width:
            scale = width / height
            new_height = self.target_size
            new_width = new_height * scale
        else:
            scale = height / width
            new_width = self.target_size
            new_height = height * scale

        inter = self.interpolation_up if self.new_shape[0] > max(height, width) else self.interpolation_down
        resized_img = cv2.resize(input_img, (new_height, new_width), interpolation=inter)
        return input_img

def get_train_image_augmentations(img_height: str, img_width: str):
    return albumentations.Compose(
        transforms=[
            albumentations.OneOf(
                [
                    albumentations.ColorJitter(
                        brightness=0.2,
                        contrast=0.17,
                        saturation=0.3,
                        hue=0.1
                    ),
                    albumentations.FancyPCA(),
                    albumentations.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                    )
                ]
            ),
            albumentations.OneOf(
                [
                    ImageIsotropicResize(
                        new_height=img_height,
                        new_width=img_width, 
                        intepolation_up=cv2.INTER_LINEAR,
                        interpolation_down=cv2.INTER_CUBIC,
                    ),
                    ImageIsotropicResize(
                        new_height=img_height,
                        new_width=img_width,
                        interpolation_up=cv2.INTER_CUBIC,
                        interpolation_down=cv2.INTER_NEAREST,
                    ),
                    ImageIsotropicResize(
                        new_height=img_height,
                        new_width=img_width,
                        interpolation_up=cv2.INTER_LINEAR,
                        interpolation_down=cv2.INTER_NEAREST,
                    ),
                ]
            ),
            albumentations.PadIfNeeded(
                min_height=img_height,
                min_width=img_width,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            albumentations.Normalize(
                mean=norm_mean,
                std=norm_std
            ),
        ]
    )

def get_val_image_augmentations(img_height: int, img_width: int):
    """
    Set of augmentations for evaluating
    image generation embedding network.
    """
    return albumentations.Compose(
        transforms=[
            albumentations.OneOf(
                [
                    ImageIsotropicResize(
                        new_height=img_height,
                        new_width=img_width, 
                        intepolation_up=cv2.INTER_LINEAR,
                        interpolation_down=cv2.INTER_CUBIC,
                    ),
                    ImageIsotropicResize(
                        new_height=img_height,
                        new_width=img_width,
                        interpolation_up=cv2.INTER_CUBIC,
                        interpolation_down=cv2.INTER_NEAREST,
                    ),
                    ImageIsotropicResize(
                        new_height=img_height,
                        new_width=img_width,
                        interpolation_up=cv2.INTER_LINEAR,
                        interpolation_down=cv2.INTER_NEAREST,
                    ),
                ]
            ),
            albumentations.PadIfNeeded(
                min_height=img_height,
                min_width=img_width,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            albumentations.Normalize(
                mean=norm_mean,
                std=norm_std
            )
        ]
    )