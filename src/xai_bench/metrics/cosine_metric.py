import numpy as np
from numpy.linalg import norm

from xai_bench.metrics.base_metric import BaseMetric
from xai_bench.metrics.explanation_normalizer import ExplanationNormalizer


class CosineMetric(BaseMetric):
    """
    Cosine distance metric between explanation vectors.

    This metric computes the angular difference between two explanation vectors.
    The vectors are optionally normalized using an ExplanationNormalizer
    (default L2 normalization), so the metric is scale-invariant.

    Parameters
    ----------
    normalization_mode : str, default="l2"
        Normalization method to apply to the explanations before computing distance.
        Options: "l2" (default), "l1", "none".

    Methods
    -------
    compute(e1: np.ndarray, e2: np.ndarray) -> float
        Returns 1 minus the cosine similarity between e1 and e2.
    """
    def __init__(self, normalization_mode="l2"):
        super().__init__("cosine")
        self.normalizer = ExplanationNormalizer(normalization_mode)

    def _compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        e1 = self.normalizer(e1)
        e2 = self.normalizer(e2)
        cosine_sim = np.dot(e1, e2)
        return 1 - cosine_sim
    

if __name__ == "__main__":
    exp_1 = np.array([0.4, 0.3, 0.2])
    exp_2 = np.array([-0.4, 0.3, 0.2])
    exp_3 = np.array([0.2, 0.3, 0.4])

    metric = CosineMetric()
    print(metric.compute(exp_1, exp_2))
    print(metric.compute(exp_1, exp_3))
    print(metric.compute(exp_2, exp_3))