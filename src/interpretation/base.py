from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    
    @abstractmethod
    def explain(self, **kwargs):
        """
        Provides explanation of the network
        behaviour, either qualitative or quanitative.
        Warning:
            this is just empty shell, which is implemented
            in other classes.
        """