from abc import ABC, abstractmethod
import numpy as np


class BaseAttack(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def generate(self, x: np.ndarray) -> np.ndarray:
        pass