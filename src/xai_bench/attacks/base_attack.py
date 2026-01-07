from abc import ABC, abstractmethod
import pandas as pd


class BaseAttack(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def generate(self, x: pd.Series) -> pd.Series:
        pass