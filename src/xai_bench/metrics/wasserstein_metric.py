import numpy as np

from xai_bench.metrics.base_metric import BaseMetric
from xai_bench.metrics.explanation_normalizer import ExplanationNormalizer


class WassersteinMetric(BaseMetric):
    """
    Wasserstein distance (Earth Mover's Distance) between explanation vectors.

    This metric treats the explanation vectors as discrete distributions and
    computes the cumulative distribution difference. The vectors are optionally
    normalized (default L1) to sum to 1.

    Parameters
    ----------
    normalization_mode : str, default="l1"
        Normalization mode applied to the explanation vectors. Default "l1" makes
        vectors sum to 1 so that Wasserstein measures relative redistribution.

    Methods
    -------
    compute(e1: np.ndarray, e2: np.ndarray) -> float
        Returns the Wasserstein distance between the two explanation vectors.
    """
    def __init__(self, normalization_mode="l1"):
        super().__init__("wasserstein")
        self.normalizer = ExplanationNormalizer(normalization_mode)

    def _compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        e1 = self.normalizer(e1)
        e2 = self.normalizer(e2)
        c1 = np.cumsum(e1)
        c2 = np.cumsum(e2)
        return np.sum(np.abs(c1 - c2))

    
if __name__ == "__main__":
    exp_1 = np.array([0.4, 0.3, 0.2])
    exp_2 = np.array([-0.4, 0.3, 0.2])
    exp_3 = np.array([0.2, 0.3, 0.4])

    metric = WassersteinMetric()
    print(metric.compute(exp_1, exp_2))
    print(metric.compute(exp_1, exp_3))
    print(metric.compute(exp_2, exp_3))
