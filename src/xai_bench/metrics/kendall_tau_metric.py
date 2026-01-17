import numpy as np
from scipy.stats import kendalltau

from xai_bench.base import BaseMetric

class KendallTauMetric(BaseMetric):
    """
    Kendall Tau distance between explanation vectors.

    The distance is computed between the ranking index vectors of the given
    explanations to sort them ascensingly. The Kendall Tau distance between
    the two rankings is determined using the `scipy.stats.kendalltau()`
    function. Since the function returns a similarity score in the range
    [-1, 1], it is converted to a distance metric in the range [0, 1], i.e.
    with interpretation [similar, unsimilar], by computing (-tau + 1) / 2
    for each pair of explanation vectors.
    """
    def __init__(self):
        super().__init__("kendall_tau")

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

        # produce rankings
        e1_ranking = np.argsort(np.abs(e1), axis=-1, stable=True)
        e2_ranking = np.argsort(np.abs(e2), axis=-1, stable=True)

        # compute kendall tau distance for each instance
        taus = np.array([
            kendalltau(ranking_1, ranking_2)[0]
            for ranking_1, ranking_2 in zip(e1_ranking, e2_ranking)             
        ])

        return (-taus + 1) / 2

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

    metric = KendallTauMetric()
    print(metric.compute(exps_1, exps_2))
    print(metric.compute(exps_1[0], exps_3))
    print(metric.compute(exps_2[1], exps_3))
