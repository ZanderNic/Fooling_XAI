# std lib imports 
from typing import Literal, Optional

# 3 party imports
import numpy as np

# projekt imports
from xai_bench.base import BaseAttack, BaseDataset, BaseModel, BaseExplainer
from xai_bench.metrics.base_metric import BaseMetric


class RandomWalkAttack(BaseAttack):
    """
        The RandomWalkAttack is a naive local attack that performs random sequential changes 
        of the features to generate a perturbed input `x_adv`. A step is only taken if 
        the distance of model predictions is bounded by epsilon. This is checked via the 
        is_attack_valid method of the base attack class.

        Supports mixed feature spaces:
        - Numerical features: small random steps within observed feature ranges.
        - Categorical features: random category switches based on training-set values.

        Parameters:
            dataset : BaseDataset
                Dataset object providing feature metadata such as:
                - numerical_features
                - categorical_features
                - feature_ranges / scaled_feature_ranges
                - categorical_values / scaled_categorical_values

            model : BaseModel
                Model to attack.

            explainer : BaseExplainer
                Explainer used to compute local explanations for individual samples.

            metric : BaseMetric
                Metric used to quantify the difference between the original explanation and the
                explanation of a candidate adversarial sample. The naive RandomWalkAttack does 
                not use the metric.

            epsilon : float, default=0.05
                Maximum allowed perturbation size (constraint handled by `is_attack_okay`).

            num_steps : int, default=100
                Number of steps.

            step_len : float, default=0.01
                Step length used for numerical feature updates (scaled by feature range).

            seed : Optional[int], default=None
                Random seed for reproducibility. If None, a random seed is used.

            task : {"classification", "regression"}, default="classification"
                Task type of the model.
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        explainer: BaseExplainer,
        metric : BaseMetric,
        epsilon: float = 0.05,
        num_steps: int = 100,
        step_len: float = 0.1,
        num_samples_explainer: float = 100,
        seed: Optional[int] = None,
        task: Literal["classification", "regression"] = "classification",
    ):
        super().__init__(model=model, task=task, epsilon=epsilon, stats=[self, "RandomWalkAttack"],dataset=dataset)
        self.explainer = explainer
        self.metric = metric
        
        self.num_steps = num_steps
        self.step_len = step_len
        self.num_samples_explainer = num_samples_explainer
        
        assert self.dataset.features is not None
        self.cols = list(self.dataset.features.feature_names_model)
        self.col2idx = {c: i for i, c in enumerate(self.cols)}

        self.n_numerical = len(self.dataset.numerical_features)
        self.n_categorical = len(self.dataset.categorical_features)

        self.numerical_features = self.dataset.numerical_features
        self.categorical_features = self.dataset.categorical_features

        self.ranges = self.dataset.scaled_feature_ranges 
        self.feature_mapping = self.dataset.feature_mapping

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)                         


    def _step(self, x: np.ndarray) -> np.ndarray:
        """
        Perform a single random walk step by perturbing exactly one feature.

        Numerical features are perturbed by a small random step within their
        observed range. Categorical features are changed by randomly selecting
        a different category observed during training.

        Parameters
        ----------
        x : np.ndarray
            Input sample of shape (d,).

        Returns
        -------
        np.ndarray
            Perturbed sample of shape (d,).
        """
        x_adv = x.copy()

        if self.rng.random() < self.n_numerical / (self.n_numerical + self.n_categorical) or self.n_categorical == 0:
            feat = self.rng.choice(self.numerical_features)
            idx = self.col2idx[feat]

            f_min, f_max = self.ranges[feat]
            span = f_max - f_min

            step = self.rng.uniform(
                -self.step_len * span,
                self.step_len * span
            )

            new_val = np.clip(x_adv[idx] + step, f_min, f_max)
            x_candidate = x_adv.copy()
            x_candidate[idx] = new_val

        else:
            if self.n_categorical == 0:
                return RuntimeError("Dataset has no features")
            
            feat = self.rng.choice(self.categorical_features)
            cols = self.feature_mapping[feat]

            idxs = [self.col2idx[c] for c in cols]

            active = [i for i in idxs if x_adv[i] == 1]

            if len(active) != 1:
                return x_adv

            current_idx = active[0]
            other_idxs = [i for i in idxs if i != current_idx]

            if not other_idxs:
                return x_adv

            new_idx = self.rng.choice(other_idxs)

            x_candidate = x_adv.copy()
            x_candidate[current_idx] = 0.0
            x_candidate[new_idx] = 1.0

        if self.is_attack_valid(x.reshape(1, -1), x_candidate.reshape(1, -1))[0]:
            return x_candidate

        return x_adv


    def fit(self) -> None:
        """
            Necessary for interface
        """
        pass 


    def _generate(self, x: np.ndarray) -> np.ndarray:
        """
            Generate an adversarial sample using the random walk.

            The method takes random steps from the origin `x` to create a pertubed sample 
            `x_adv` that fulfills the epsilon contraint.

            Parameters 
                x : np.ndarray
                    Original input sample of shape (d,).

            Returns
                np.ndarray
                    Adversarial sample found (shape (d,)).
        """ 
        for _ in range(self.num_steps):
            x = self._step(x)

        return x