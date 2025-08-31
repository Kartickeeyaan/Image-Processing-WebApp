# cvops/base.py
from abc import ABC, abstractmethod

class Operation(ABC):
    name = "BaseOperation"
    category = "Base"

    @abstractmethod
    def apply(self, img, **kwargs):
        """Apply the operation to a BGR uint8 image and return a BGR uint8 result."""
        raise NotImplementedError
