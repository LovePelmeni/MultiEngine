from torch.utils.tensorboard.writer import SummaryWriter
from src.training.callbacks import base
import typing

class AccuracyCallback(base.BaseCallback):
    """
    Module callback for storing training 
    performance accuracy of the Similarity Search algorithm.

    Parameters:
    ------------
        report_path - path to save reports
        perf_callbacks - performance callbacks
    """
    def __init__(self, log_writer: SummaryWriter):
        self.perf_callbacks = log_writer
        self.log_writer = log_writer 

    def save_report(self, **kwargs):
        accuracy = kwargs.get('accuracy')
        global_step = kwargs.get('global_step')
        self.log_writer.add_scalar(
            tag='search_index_accuracy',
            scalar_value=accuracy,
            global_step=global_step
        )