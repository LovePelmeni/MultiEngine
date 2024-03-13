import typing
import pathlib
import cv2
import numpy
import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename="video_utils_logs.log")
logger.addHandler(file_handler)

def get_video_numpy_array(video_url: typing.Union[str, pathlib.Path]) -> numpy.ndarray:
    """
    Extracts video frames
    and converts them to numpy.ndarray object.
    """
    try:
        video = cv2.VideoCapture(video_url)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_ch = int(video.get(cv2.CAP_PROP_CHANNEL))

        output_frames = numpy.empty((total_frames, frame_height, frame_width, frame_ch))
        curr_idx = 0

        while curr_idx < total_frames:
            success, output_frames[curr_idx] = video.read()

            if success == False:
                print("failed to extract frame at index: %s" % curr_idx)
            curr_idx = curr_idx + 1
        return output_frames
    except(Exception) as err:
        logger.error(err)
        
def convert_video_rgb(videos: typing.List[numpy.ndarray]) -> numpy.ndarray:
    """
    Converts video frames to 
    RGB color scheme.
    """
    for idx, video_frame in enumerate(videos):
        videos[idx] = cv2.cvtColor(video_frame, cv2.IMREAD_UNCHANGED)
    return videos