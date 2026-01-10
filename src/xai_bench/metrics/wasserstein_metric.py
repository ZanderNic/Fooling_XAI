import numpy as np
from scipy.stats import wasserstein_distance

from xai_bench.metrics.base_metric import BaseMetric


class WassersteinMetric(BaseMetric):
    def __init__(self):
        super().__init__("wasserstein")

    def _wd(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.sum() <= 0 or b.sum() <= 0:
            return 0.0

        a = a / a.sum()
        b = b / b.sum()

        positions = np.arange(len(a))
        return wasserstein_distance(positions, positions, a, b)

    def compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        p1 = np.clip(e1, 0, None)
        p2 = np.clip(e2, 0, None)

        n1 = np.clip(-e1, 0, None)
        n2 = np.clip(-e2, 0, None)

        return self._wd(p1, p2) + self._wd(n1, n2)
    

if __name__ == "__main__":
    exp_1 = np.array([0.4, 0.3, 0.2])
    exp_2 = np.array([-0.4, 0.3, 0.2])
    exp_3 = np.array([0.2, 0.3, 0.4])

    metric = WassersteinMetric()
    print(metric.compute(exp_1, exp_2))
    print(metric.compute(exp_1, exp_3))
    print(metric.compute(exp_2, exp_3))
