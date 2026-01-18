import numpy as np

from xai_bench.metrics.base_metric import BaseMetric
from xai_bench.metrics.explanation_normalizer import ExplanationNormalizer


class L2Metric(BaseMetric):
    """
    L2 (Euclidean) distance between explanation vectors.

    The distance is optionally normalized by the number of features to
    allow comparison across datasets with different feature dimensionality.

    Parameters
    ----------
    normalization_mode : str, default="l2"
        Normalization method to apply to the explanations before computing distance.
        Options: "l2" (default), "l1", "none".

    Methods
    -------
    compute(e1: np.ndarray, e2: np.ndarray) -> np.ndarray
        Returns an array of L2 distances for each explanation vector pair.
    """
    def __init__(self):
        super().__init__("l2")
        self.normalizer = ExplanationNormalizer(mode="l2")

    def _compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        raise NotImplementedError("This is wrong. use .compute instead")
        e1 = self.normalizer(e1)
        e2 = self.normalizer(e2)
        return np.linalg.norm(e1 - e2)

    def compute(self, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
        e1 = np.asarray(e1)
        e2 = np.asarray(e2)

        if e1.ndim == 1:
            e1 = e1[None, :]
        if e2.ndim == 1:
            e2 = e2[None, :]

        e1 = self.normalizer(e1)
        e2 = self.normalizer(e2)

        diff = e1 - e2

        return np.linalg.norm(diff, axis=1)

if __name__ == "__main__":
    exp_1 = np.array([1, 0, 0])
    exp_2 = np.array([-1, 0, 0])
    exp_3 = np.array([0, 1, 0])

    metric = L2Metric()
    print(metric.compute(exp_1, exp_1))
    print(metric.compute(exp_1, exp_2))
    print(metric.compute(exp_1, exp_3))

    print(metric._compute(exp_1, exp_1))
    print(metric._compute(exp_1, exp_2))
    print(metric._compute(exp_1, exp_3))
