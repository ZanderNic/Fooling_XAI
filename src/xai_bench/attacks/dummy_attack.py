# std lib imports 
from typing import Literal, Optional

# 3 party imports
import numpy as np

# projekt imports
from xai_bench.base import BaseAttack, BaseDataset, BaseModel, BaseExplainer
from xai_bench.metrics.base_metric import BaseMetric


class DummyAttack(BaseAttack):
    """
    Returns the input.
    """
    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        explainer: BaseExplainer,
        metric : BaseMetric,
        epsilon: float = 0.05,
        task: Literal["classification", "regression"] = "classification",
    ):
        super().__init__(model=model, task=task, epsilon=epsilon, stats=[self, "MonteCarloAttack"],dataset=dataset)
        self.explainer = explainer
        self.metric = metric

    def fit(self):
        # Interface
        pass

    def _generate(self, x: np.ndarray) -> np.ndarray:
        # Returns the input
        return x