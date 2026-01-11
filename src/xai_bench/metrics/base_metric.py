import numpy as np
from abc import ABC, abstractmethod

class BaseMetric(ABC):
    def __init__(self, name: str):
        self.name = name

    # should be a distance, so maximising
    @abstractmethod
    def compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        pass