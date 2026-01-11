from abc import ABC, abstractmethod
import numpy as np

from xai_bench.base import BaseModel
from typing import Literal, overload
from numbers import Number
import pandas as pd

class BaseAttack(ABC):
    def __init__(self, model: BaseModel, task: Literal["classification","regression"]):
        self.model = model
        self.task: Literal["classification","regression"] = task

    """
    Will recieve ONE sample to generate an attack on
    """
    @abstractmethod
    def generate(self, x: np.ndarray) -> np.ndarray:
        pass

    """
    Will calcualte L1 distance and return mean distance on dataset
    """
    @overload
    def _prediction_distance(self, X:np.ndarray, X_adv:np.ndarray) -> float:
        pass
    @overload
    def _prediction_distance(self, X:pd.DataFrame, X_adv:pd.DataFrame) -> float:
        pass
    def _prediction_distance(self, X, X_adv):
        if self.task == "classification":
            p: np.ndarray = self.model.predict_proba(X)
            p_adv: np.ndarray = self.model.predict_proba(X_adv)
        else:
            p: np.ndarray = self.model.predict_scalar(X)
            p_adv: np.ndarray = self.model.predict_scalar(X_adv)
        return np.abs(p - p_adv).mean()