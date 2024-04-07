from src.training.callbacks import base
from torch.utils.tensorboard.writer import SummaryWriter
import typing
import pathlib
import os

class StorageCallback(base.BaseCallback):
    """
    Measures the weight of the search index
    after it has been trained.
    """
    def __init__(self, 
        log_writer: SummaryWriter, 
        output_index_file_path: typing.Union[str, pathlib.Path]
    ):
        self.output_index_path = output_index_file_path 
        self.log_writer = log_writer

    def on_validation_end(self):
        self.total_bytes = os.path.getsize(
        filename=self.output_index_path)
        output_megabytes = self.writer.add_scalar(
            tag="search_index_capacity_disk",
            scalar_value=output_megabytes,
            global_step=output_megabytes
        )



