from torch.utils.tensorboard.writer import SummaryWriter
from src.training.callbacks import base
import typing

class ReportCallback(base.BaseCallback):
    """
    Module callback for storing training 
    performance metrics of the Similarity Search algorithm.

    Parameters:
    ------------
        report_path - path to save reports
        perf_callbacks - performance callbacks
    """
    def __init__(self, 
        report_path: str, 
        perf_callbacks: typing.List[base.BaseCallback],
        faiss_hyperparams: typing.Dict[str, typing.Any]
    ):
        self.perf_callbacks = perf_callbacks
        self.writer = SummaryWriter(log_dir=report_path)
        self.faiss_hyperparams = faiss_hyperparams

    def save_report(self):
        pass
    