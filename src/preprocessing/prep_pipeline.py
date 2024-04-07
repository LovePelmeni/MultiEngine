import logging
import argparse
from torch.utils import data
from src.training.face_detection import face_detection
from src.training import video_utils
from src.preprocessing import (
    image_augmentations as img_aug,
    audio_augmentations as aud_aug,
    text_augmentations as text_aug
)
import pathlib
import torch
import os
import json 
from glob import glob

runtime_logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename='prep_pipeline_runtime.log')
runtime_logger.addHandler(file_handler)

def main():
    parser = argparse.ArgumentParser(description='Pipeline for generating face crops from dialog videos')
    data_group = parser.add_argument_group("data source information")
    inf_group = parser.add_mutually_exclusive_group("inference hardware")

    data_group.add_argument(
        "--video-source-path", 
        dest='video_source_path',
        type=str, required=True, 
        help='folder, where video data is stored'
    )
    data_group.add_argument(
        "--json-config-path",
        dest='json_config_path',
        help='path to json configuration file with preprocessing setup'
    )
    data_group.add_argument(
        "--type", dest='data_type', choices=['train', 'valid'],
        required=True, help='type of the dataset to process'
    )
    inf_group.add_argument(
        "--use-cpu", action=False,
        help='use cpu to run preprocessing job'
    )
    inf_group.add_argument(
        "--use-gpu", action=True, 
        help='use gpu to run preprocessing job'
    )
    inf_group.add_argument(
        "--use-mps", action=True, 
        help='use mps backend to run preprocessing job'
    )
    parser.add_argument(
        "--log-dir", required=False, 
        default='logs', help='path to store logs of preprocessing'
    )

    args = parser.parse_args()
    vid_source_path = pathlib.Path(args.video_source_path)
    json_config_path = pathlib.Path(args.json_config_path)
    data_type = args.data_type
    log_dir = pathlib.Path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    inference_device = torch.device('cpu')

    config_parent = json_config_path.parent
    config_filename = json_config_path.name

    config_file = json.load(
        glob(recursive=True, 
        pathname=config_parent + "/*/" + config_filename
        )
    )

    # dataset characteristics 
    resize_face_height = config_file.get("resize_face_height")
    resize_face_width = config_file.get("resize_face_width")
    face_margin = config_file.get("face_margin")
    min_face_size = config_file.get("min_face_size")
    input_img_size = config_file.get("input_img_size")
    
    # hardware loading settings
    loader_num_workers = config_file.get("loader_num_workers", 1)
    
    # video characteristics
    ith_pick_frame = config_file.get("ith_frame_per_video")

    video_paths = [
        os.path.join(vid_source_path, vid_name)
        for vid_name in os.listdir(vid_source_path)
    ]

    face_detector = face_detection.HumanFaceDetector(
        input_img_size=input_img_size,
        min_face_size=min_face_size,
        face_margin=face_margin,
        inference_device=inference_device
    )
    
    if (args.use_gpu == True):
        inference_device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True 
        # in case data has the same size, it will turn on additional optimization 
        # should be disabled during actual inference, as may introduce additional
        # overload

    if data_type == 'train':
        face_augmentations = img_aug.get_train_image_augmentations(
            img_height=resize_face_height, 
            img_width=resize_face_width, 
            use_normalization=False
        )
        audio_augmentations = aud_aug.get_train_audio_augmentations()
        text_augmentations = text_aug.get_train_text_augmentations()
    
    if data_type == 'valid':
        face_augmentations = img_aug.get_val_image_augmentations(
            img_height=resize_face_height, 
            img_width=resize_face_width,
            use_normalization=False
        )
        audio_augmentations = aud_aug.get_val_audio_augmentations()
        text_augmentations = text_aug.get_val_text_augmentations()

    modal_dataset = video_utils.VideoDataset(
        video_paths=video_paths,
        pick_ith_frame=ith_pick_frame,
    )

    loader = data.DataLoader(
        dataset=modal_dataset,
        batch_size=1,
        num_workers=loader_num_workers,
        shuffle=False,
    )
    
    try:
        for frame, audio_signal, text in loader:
            
            augmented_frame = face_augmentations(image=frame)['image']
            img_height, img_width = frame.shape[:2]
            output_boxes = []

            # processing human faces detected bounding boxes 
            face_boxes, _ = face_detector.detect_faces(
                input_img=augmented_frame, 
                use_landmarks=False
            )

            for (x1, y1, x2, y2) in face_boxes:
                x1 = min(max(x1, 0), img_width-1)
                y1 = min(max(y1, 0), img_height-1)
                x2 = min(max(x2, 0), img_width-1)
                y2 = min(max(y2, 0), img_height-1)
                output_boxes.append(augmented_frame[y1:y2, x1:x2])
            
            # processing audio
            augmented_signal = audio_augmentations(audio_signal)['audio']

            # processing text
            augmented_text = text_augmentations()['text']
            
    except(Exception) as err:
        runtime_logger.error(err)
        raise SystemExit("Pipeline ended up breaking, due to some error, check logs.")
            
            