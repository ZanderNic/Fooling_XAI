import numpy as np
from numpy.linalg import norm

from xai_bench.metrics.base_metric import BaseMetric


class CosineMetric(BaseMetric):
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

    metric = CosineMetric()
    print(metric.compute(exp_1, exp_2))
    print(metric.compute(exp_1, exp_3))
    print(metric.compute(exp_2, exp_3))