import abc
import torch
import logging 

logger = logging.getLogger(__name__)
handler = logging.FileHandler(filename='quantizer.log')
logger.addHandler(handler)

class BaseQuantizer(abc.ABC):
    """
    Base module for quantizing
    parameters of a network or input.
    """
    def __init__(self, quan_type: torch.dtype):
        self.quan_type = quan_type
    
    @abc.abstractmethod
    def quantize(self, **kwargs):
        """
        Quantizes parameters, based on the input
        arguments.
        """
