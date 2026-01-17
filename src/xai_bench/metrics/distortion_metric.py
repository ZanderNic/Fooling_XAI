import numpy as np

from xai_bench.metrics.explanation_normalizer import ExplanationNormalizer

from xai_bench.base import BaseMetric
from xai_bench.metrics.kendall_tau_metric import KendallTauMetric

class DistortionMetric(BaseMetric):
    """
    Explanation distortion metric combining L1 distance and Kendall Tau distance.

    The metric computes a weighted sum of the L1 distance between two explanation
    vectors and the Kendall Tau distance between their rankings. The explanations
    are first normalized using L1 normalization before computing the distances.
    """
    def __init__(
        self, 
        weights: tuple[float, float] = (0.5, 0.5)
    ):
        super().__init__("distortion_(L1+KendallTau)")
        self.normalizer = ExplanationNormalizer(mode="l1")
        self.kendall_tau_metric = KendallTauMetric()

        if isinstance(weights, (list, tuple)) and \
            len(weights) == 2 and sum(weights) == 1:
            self.weights = weights
        else:
            raise ValueError(
                "Weights must be a list or tuple of two numbers summing to 1."
            )

    def _compute(
        self,
        e1: np.ndarray,
        e2: np.ndarray
    ) -> np.ndarray:
        return self.compute(e1, e2)

    def compute(
        self,
        e1: np.ndarray,
        e2: np.ndarray
    ) -> np.ndarray:
        assert e1.shape == e2.shape, "Explanation vectors must have the same shape."
        assert 1 <= e1.ndim <= 2, "Explanation vectors must be 1D or 2D arrays."

        if e1.ndim == 1:
            e1 = e1.reshape(1, -1)
        if e2.ndim == 1:
            e2 = e2.reshape(1, -1)

        e1 = self.normalizer(e1)
        e2 = self.normalizer(e2)

        # compute the L1 distance
        l1_distance = np.sum(np.abs(e1 - e2), axis=-1)

        # compute kendall tau distance
        kendall_tau_distance = self.kendall_tau_metric.compute(e1, e2)

        # combine both distances
        combined_distance = (
            self.weights[0] * l1_distance +
            self.weights[1] * kendall_tau_distance
        )

        return combined_distance

if __name__ == "__main__":
    exps_1 = np.array([
        [0.4, 0.3, 0.2],
        [-0.2, 0.1, 0.4],
        [0.3, -0.4, 0.2]
    ])
    exps_2 = np.array([
        [-0.4, 0.3, 0.2],
        [-0.1, 0.2, 0.4],
        [0.2, -0.3, 0.4]
    ])
    exps_3 = np.array([0.2, 0.3, 0.4])

    metric = DistortionMetric()
    print(metric.compute(exps_1, exps_2))
    print(metric.compute(exps_1[0], exps_3))
    print(metric.compute(exps_2[1], exps_3))
