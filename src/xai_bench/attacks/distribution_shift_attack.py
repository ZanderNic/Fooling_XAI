import numpy as np
import pandas as pd

from xai_bench.attacks.base_attack import BaseAttack


class DistributionShiftAttack(BaseAttack):
    def __init__(
        self,
        dataset,
        model,
        epsilon: float = 0.1,
        max_tries: int = 100,
        prob_tolerance: float = 0.01
    ):
        super().__init__(model)
        self.dataset = dataset
        self.model = model
        self.epsilon = epsilon
        self.max_tries = max_tries
        self.prob_tolerance = prob_tolerance

        self.feature_ranges = self.dataset.feature_ranges
        self.protected_features = self.dataset.categorical_features

    def _prediction_distance(self, x, x_adv):
        p = self.model.predict_proba(pd.DataFrame([x]))[0]
        p_adv = self.model.predict_proba(pd.DataFrame([x_adv]))[0]
        return np.abs(p - p_adv).max()
    
    def _shift_feature(self, x, feature):
        x_new = x.copy()
        f_min, f_max = self.feature_ranges[feature]
        span = f_max - f_min

        shift = np.random.uniform(-self.epsilon, self.epsilon) * span
        x_new[feature] = np.clip(x[feature] + shift, f_min, f_max)
        return x_new

    def generate(self, x: np.ndarray) -> np.ndarray:
        x_adv = x.copy()
        candidates = [
            f for f in x.index
            if f not in self.protected_features
            and f in self.feature_ranges
        ]

        for _ in range(self.max_tries):
            feature = np.random.choice(candidates)
            x_candidate = self._shift_feature(x_adv, feature)
            prob_shift = self._prediction_distance(x, x_candidate)

            if prob_shift <= self.prob_tolerance:
                x_adv = x_candidate

        return x_adv