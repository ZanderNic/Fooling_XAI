# std-lib imports
from typing import cast

# 3 party imports
import numpy as np
from scipy.stats import spearmanr

# projekt imports
from xai_bench.metrics.base_metric import BaseMetric
from xai_bench.metrics.explanation_normalizer import ExplanationNormalizer


class SpearmanMetric(BaseMetric):
    """
    Spearman rank correlation distance between explanation vectors.

    This metric measures the dissimilarity between the rankings of feature importances
    in two explanations. It ignores absolute values and only considers order.

    Parameters
    ----------
    normalizer_mode : str, default="none"
        Normalization mode applied to the vectors. Usually "none" since ranking is scale-invariant.

    Methods
    -------
    compute(e1: np.ndarray, e2: np.ndarray) -> float
        Returns 1 minus the Spearman rank correlation between e1 and e2.
        Result ranges from 0 (identical ranking) to 2 (opposite ranking).
    """
    def __init__(self, normalizer_mode="none"):
        super().__init__("spearman")
        self.normalizer = ExplanationNormalizer(normalizer_mode)

    def _compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        e1 = self.normalizer(e1)
        e2 = self.normalizer(e2)
        r1 = np.argsort(-np.abs(e1))
        r2 = np.argsort(-np.abs(e2))
        res = spearmanr(r1, r2)
        return 1 - cast(float, res[0])
    

if __name__ == "__main__":
    exp_1 = np.array([0.4, 0.3, 0.2])
    exp_2 = np.array([-0.4, 0.3, 0.2])
    exp_3 = np.array([0.2, 0.3, 0.4])

    metric = SpearmanMetric()
    print(metric.compute(exp_1, exp_2))
    print(metric.compute(exp_1, exp_3))
    print(metric.compute(exp_2, exp_3))

