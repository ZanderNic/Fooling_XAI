import numpy as np
from scipy.stats import spearmanr

from xai_bench.metrics.base_metric import BaseMetric


class SpearmanMetric(BaseMetric):
    def __init__(self):
        super().__init__("spearman")

    def compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        r1 = np.argsort(-np.abs(e1))
        r2 = np.argsort(-np.abs(e2))
        r, _ = spearmanr(r1, r2)
        return 1 - r
    

if __name__ == "__main__":
    exp_1 = np.array([0.4, 0.3, 0.2])
    exp_2 = np.array([-0.4, 0.3, 0.2])
    exp_3 = np.array([0.2, 0.3, 0.4])

    metric = SpearmanMetric()
    print(metric.compute(exp_1, exp_2))
    print(metric.compute(exp_1, exp_3))
    print(metric.compute(exp_2, exp_3))

