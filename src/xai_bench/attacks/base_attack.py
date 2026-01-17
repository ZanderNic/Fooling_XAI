from abc import ABC, abstractmethod
import numpy as np

from xai_bench.base import BaseModel, BaseDataset
from typing import Literal, overload, Optional
import pandas as pd
from xai_bench.stat_collector import StatCollector
from rich.progress import Progress, TextColumn,BarColumn,TimeElapsedColumn,TimeRemainingColumn
from xai_bench.console import console, RUN_TEXT

class BaseAttack(ABC):
    def __init__(self, model: BaseModel, task: Literal["classification","regression"], epsilon:Optional[float], stats, dataset:BaseDataset):
        self.model = model
        self.task: Literal["classification","regression"] = task
        self.epsilon: Optional[float] = epsilon
        self.stats = StatCollector(obj=stats[0],comment=stats[1])
        self.dataset = dataset
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
            with self._get_generate_progress_bar() as progress:
                task = progress.add_task(f"{RUN_TEXT} Generating Attack", total=len(x))
                samples = []
                for s in x:
                    samples.append(self._generate(s))
                    progress.update(task,advance=1)
                return np.asarray(samples)
        else:
            return self._generate(x)

    @abstractmethod
    def _generate(self, x) -> np.ndarray:
        pass

    """
    Will calculate L1 distance and return mean distance on dataset
    """
    @overload
    def _prediction_distance(self, X:np.ndarray, X_adv:np.ndarray) -> np.ndarray:
        pass
    @overload
    def _prediction_distance(self, X:pd.DataFrame, X_adv:pd.DataFrame) -> np.ndarray:
        pass
    def _prediction_distance(self, X, X_adv):
        assert (isinstance(X,pd.DataFrame)) or (isinstance(X,np.ndarray) and X.ndim==2), "Dataframe or Single sample with shape (1,num_features)"
        assert (isinstance(X_adv,pd.DataFrame)) or (isinstance(X_adv,np.ndarray) and X.ndim==2), "Dataframe or Single sample with shape (1,num_features)"
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
    def is_attack_valid(self, X:pd.DataFrame, X_adv:pd.DataFrame, epsilon:Optional[float]=None) -> tuple[np.ndarray, int]:
        pass
    def is_attack_valid(self, X, X_adv, epsilon=None):
        # prediction distance
        p_dist = self._prediction_distance(X,X_adv)

        if p_dist.ndim == 1:
            p_dist = p_dist.reshape(-1,1)

        if self.task=="regression":
            # scale to [0-1] in case of regression
            p_dist = (p_dist-self.dataset.y_range["max"])/(self.dataset.y_range["max"]/self.dataset.y_range["min"])
        # epsilon
        if epsilon is None:
            assert self.epsilon is not None, "At least one epsilon must be present"
            epsilon = self.epsilon
        okays:np.ndarray = p_dist <= epsilon
        # return ob okay; num_feature/num_samples okay
        return okays.all(axis=-1), okays.sum(axis=-1)
    
    """
    Will use attack when valid, otherwise original
    """
    def wrap_attack_valid(self, X, X_adv, epsilon:Optional[float]=None):
        _, sums = self.is_attack_valid(X,X_adv,epsilon)
        return np.where(sums,X_adv,X)
    
    def _get_generate_progress_bar(self):
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.completed}/{task.total}",
            "[progress.percentage]({task.percentage:>3.1f}%)",
            "[bold #ed1cdf]•[/]",
            TimeElapsedColumn(),
            "[bold #ed1cdf]/[/]",
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )