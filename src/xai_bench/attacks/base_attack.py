from abc import ABC, abstractmethod
import numpy as np

from xai_bench.base import BaseModel


class BaseAttack(ABC):
    def __init__(self, model: BaseModel):
        self.model = model

    """
    Will recieve ONE sample to generate an attack on
    """
    @abstractmethod
    def generate(self, x: np.ndarray) -> np.ndarray:
        pass