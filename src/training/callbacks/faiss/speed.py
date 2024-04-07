from src.training.callbacks import base
from torch.utils.tensorboard.writer import SummaryWriter
import time

class InferenceSpeedCallback(base.BaseCallback):
    """
    Module callback for measuring performance
    speed during seach index training.
    """
    def __init__(self, logistics_writer: SummaryWriter):
        self.log_writer = logistics_writer
    
    def on_validation_start(self):
        self.starter = time.perf_counter()

    def on_validation_end(self, **kwargs):
        self.ender = time.perf_counter()
        distance = self.ender - self.starter
        global_step = kwargs.get("global_step")
        self.log_writer.add_scalar(
            tag="search_speed", 
            scalar_value=distance,
            global_step=global_step
        )