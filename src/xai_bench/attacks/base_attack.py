from abc import ABC, abstractmethod
import numpy as np

from xai_bench.base import BaseModel
from typing import Literal, overload, Optional
from numbers import Number
import pandas as pd
from jaxtyping import Shaped
from xai_bench.stat_collector import StatCollector

class BaseAttack(ABC):
    def __init__(self, model: BaseModel, task: Literal["classification","regression"], epislon:Optional[float]):
        self.model = model
        self.task: Literal["classification","regression"] = task
        self.epsilon: Optional[float] = epislon
        self.stats = StatCollector(obj=self,comment="Calls of attack")
    """
    Call beforehand in order to setup the attack. (e.g. finding best parameters)
    """
    @abstractmethod
    def fit(self):
        pass

    """
    Will recieve either ONE sample of shape (features,) or multiple samples of shape (n,features) to generate an attack on
    
    Every sample that gets returned has to pass the is_attack_okay function. Meaning, all samples prediction threshhold must remain <= epsilon. For every sample that doenst pass the thresholding, just return the original sample.
    """
    def generate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        self.stats("generate", x)
        if x.ndim == 2:
            return np.asarray([self._generate(s) for s in x])
        else:
            return self._generate(x)

    @abstractmethod
    def _generate(self, x) -> np.ndarray:
        pass

    """
    Will calcualte L1 distance and return mean distance on dataset
    """
    @overload
    def _prediction_distance(self, X:np.ndarray, X_adv:np.ndarray) -> np.ndarray:
        pass
    @overload
    def _prediction_distance(self, X:pd.DataFrame, X_adv:pd.DataFrame) -> np.ndarray:
        pass
    def _prediction_distance(self, X, X_adv):
        if self.task == "classification":
            p: np.ndarray = self.model.predict_proba(X)
            p_adv: np.ndarray = self.model.predict_proba(X_adv)
        else:
            p: np.ndarray = self.model.predict_scalar(X)
            p_adv: np.ndarray = self.model.predict_scalar(X_adv)
        return np.abs(p - p_adv)
    
    @overload
    def is_attack_valid(self, X:np.ndarray, X_adv:np.ndarray, epsilon:Optional[float]=None) -> tuple[bool, int]:
        pass
    @overload
    def is_attack_valid(self, X:pd.DataFrame, X_adv:pd.DataFrame, epsilon:Optional[float]=None) -> tuple[bool, int]:
        pass
    def is_attack_valid(self, X, X_adv, epsilon=None):
        # prediction distance
        p_dist = self._prediction_distance(X,X_adv)
        # epsilon
        if epsilon is None:
            assert self.epsilon is not None, "At least one epsilon must be present"
            epsilon = self.epsilon
        okays:np.ndarray = p_dist<=epsilon
        # return ob okay; num_feature/num_samples okay
        return okays.all(axis=-1), okays.sum(axis=-1)