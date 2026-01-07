import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import spearmanr
from scipy.stats import wasserstein_distance
import re


def parse_feature_name(lime_feature: str) -> str:
    """
    Extract base feature name from LIME condition string.
    Examples:
      'age > 50'        -> 'age'
      'thal=3'          -> 'thal'
      '0.00 < sex <= 1' -> 'sex'
    """
    tokens = re.split(r"[<>=]", lime_feature)
    for token in tokens:
        token = token.strip()
        if token.isalpha():
            return token
    return lime_feature.split()[0]


def explanation_to_array(exp, feature_names):
    arr = np.zeros(len(feature_names))

    for feat, weight in exp.as_list():
        base_feat = parse_feature_name(feat)
        if base_feat in feature_names:
            idx = feature_names.index(base_feat)
            arr[idx] += weight
    return arr


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, exp, exp_adv) -> float:
        pass


class SpearmanMetric(Metric):
    def __init__(self, feature_names):
        super().__init__("spearman")
        self.feature_names = feature_names

    def compute(self, exp, exp_adv) -> float:
        e1 = explanation_to_array(exp, self.feature_names)
        e2 = explanation_to_array(exp_adv, self.feature_names)

        r, _ = spearmanr(
            np.argsort(-np.abs(e1)),
            np.argsort(-np.abs(e2))
        )

        return 1 - r
    

class WassersteinMetric(Metric):
    def __init__(self, feature_names):
        super().__init__("wasserstein")
        self.feature_names = feature_names

    def compute(self, exp, exp_adv):
        e1 = explanation_to_array(exp, self.feature_names)
        e2 = explanation_to_array(exp_adv, self.feature_names)

        def split(e):
            pos = np.clip(e, 0, None)
            neg = np.clip(-e, 0, None)
            return pos, neg

        def wd(a, b):
            if a.sum() == 0 or b.sum() == 0:
                return 0.0
            a = a / a.sum()
            b = b / b.sum()
            pos = np.arange(len(a))
            return wasserstein_distance(pos, pos, a, b)

        p1, n1 = split(e1)
        p2, n2 = split(e2)

        return wd(p1, p2) + wd(n1, n2)
    

if __name__ == "__main__":
    class MockExplanation:
        def __init__(self, items):
            self._items = items

        def as_list(self):
            return self._items
        
    exp_1 = MockExplanation([
        ("1", 0.4),
        ("2", 0.3),
        ("3", 0.2)
    ])

    exp_2 = MockExplanation([
        ("1", -0.4),
        ("2", 0.3),
        ("3", 0.2)
    ])

    feature_names = ["1", "2", "3"]

    metric = SpearmanMetric(feature_names)
    print(metric.compute(exp_1, exp_2))

    metric = WassersteinMetric(feature_names)
    print(metric.compute(exp_1, exp_2))
