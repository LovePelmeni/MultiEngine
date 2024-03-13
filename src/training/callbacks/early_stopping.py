from types import NotImplementedType
from src.training.callbacks import base


class EarlyStoppingCallback(base.BaseCallback):
    """
    Early stopping regularization strategy
    for shutting down training phase in 
    case of performance degrading.

    Parameters:
    -----------
        min_diff - minimum difference between adjacent metric values.
        patience - number of degrading epochs to tolerate.
    """
    def __init__(self, min_diff: float, patience: int):
        self.curr_patience = patience 
        self.default_patience = patience 
        self.min_diff: float = min_diff
        self.prev_metric = None 

    def on_train_batch_end(self, **kwargs):

        trainer = kwargs.get("trainer")
        validation_dataset = kwargs.get("validation_dataset")
        curr_metric = trainer.evaluate(validation_dataset)

        if self.prev_metric is None:
            self.prev_metric = curr_metric 

        elif curr_metric - self.prev_metric < self.min_diff:
            self.curr_patience -= 1
        
        else:
            self.curr_patience = self.default_patience
        
        if self.curr_patience == 0:
            if hasattr(kwargs['trainer'], 'stop_flag'):
                kwargs['trainer'].stop_flag = True
            else:
                raise NotImplementedError()