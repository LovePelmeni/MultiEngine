from src.training.trainers import base 
from src.training.callbacks import base as callbase
from torch import nn


class ClassifierTrainer(base.BaseTrainer):
    """
    Training pipeline for MLP classification 
    network.
    
    Parameters:
    -----------
    """
    def __init__(self,):
        super(ClassifierTrainer, self).__init__()
    
