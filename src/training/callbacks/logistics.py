from src.training.callbacks import base 
import typing 
import pathlib
import torch
import os

class LogisticsCallback(base.BaseCallback):
    """
    Callback for tracking logistics of the network 
    during training process. Epoch summarization, report 
    common training information, such as loss, batch_size, etc...
    """
    def __init__(self, log_dir: typing.Union[str, pathlib.Path]):
        super(LogisticsCallback, self).__init__(log_dir=log_dir)

    def save_report(self, input_info: typing.Dict, global_step: int):
        """
        Function saves report of the training
        summarization info.
        """
        torch.save(
            obj=input_info,
            f=os.path.join(self.log_dir, "report_%s.pth" % global_step)
        )

    def on_train_end(self, **kwargs):
        report = kwargs.get('report')
        self.save_report(input_info=report)
        