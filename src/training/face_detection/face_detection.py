from facenet_pytorch import MTCNN 
import numpy
from src.preprocessing.image_augmentations import resize

class HumanFaceDetector(object):
    """
    Class for detecting human faces on the image
    using pretrained Multi Cascaded Neural Network.
    
    Parameters:
    -----------
        input_img_size - size of the input image
        face_margin - margin in pixels to add when cropping out detected face from image scene
        min_face_size - minimum size of the face that can be detected by the detector
    """
    def __init__(self, 
        input_img_size: int, 
        min_face_size: int = 160, 
        face_margin: int = 0,
        inference_device: str = 'cpu'
    ):
        self.resizer = resize.IsotropicResize(
            input_height=input_img_size,
            input_width=input_img_size
        )
        self.detector = MTCNN(
            image_size=input_img_size,
            margin=face_margin,
            min_face_size=min_face_size,
            thresholds=[80, 90, 90],
            device=inference_device
        )

    def detect_faces(self, input_img: numpy.ndarray):
        resized_img = self.resizer(input_img)['image']
        pred_boxes, _, _ = self.detector.detect(resized_img)
        img_height, img_width = resized_img.shape[:2]
        output_boxes = [0] * len(pred_boxes)
        for box in pred_boxes:
            output_boxes[box] = [
                min(max(pred_boxes[box][0], 0), img_width-1),
                min(max(pred_boxes[box][1], 0), img_height-1),
                min(max(pred_boxes[box][2], 0), img_width-1),
                min(max(pred_boxes[box][3], 0), img_height-1)
            ]
        return output_boxes