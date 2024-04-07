from abc import ABC, abstractmethod

class BaseAugmentation(ABC):
    """
    Base Abstract class for input augmentation.
    """
    @abstractmethod
    def apply(self, **kwargs):
        """
        Applies given augmentation to a set of input
        data.
        Warning:
            This is just empty shell that is implemented
            in other classes.
        """