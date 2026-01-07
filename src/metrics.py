import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import spearmanr
from scipy.stats import wasserstein_distance
from numpy.linalg import norm


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        pass


class SpearmanMetric(Metric):
    def __init__(self):
        super().__init__("spearman")

    def compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        r1 = np.argsort(-np.abs(e1))
        r2 = np.argsort(-np.abs(e2))
        r, _ = spearmanr(r1, r2)
        return 1 - r
    

class WassersteinMetric(Metric):
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


class CosineDistanceMetric(Metric):
    def __init__(self):
        super().__init__("cosine")

    def compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        if norm(e1) == 0 or norm(e2) == 0:
            return 0.0

        cosine_sim = np.dot(e1, e2) / (norm(e1) * norm(e2))
        return 1 - cosine_sim
    

if __name__ == "__main__":
    exp_1 = np.array([0.4, 0.3, 0.2])
    exp_2 = np.array([-0.4, 0.3, 0.2])
    exp_3 = np.array([0.2, 0.3, 0.4])

    metric = SpearmanMetric()
    print(metric.compute(exp_1, exp_2))

    metric = WassersteinMetric()
    print(metric.compute(exp_1, exp_2))

    metric = CosineDistanceMetric()
    print(metric.compute(exp_1, exp_2))

    metric = SpearmanMetric()
    print(metric.compute(exp_1, exp_3))

    metric = WassersteinMetric()
    print(metric.compute(exp_1, exp_3))

    metric = CosineDistanceMetric()
    print(metric.compute(exp_1, exp_3))