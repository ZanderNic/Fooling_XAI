# std lib imports 
from typing import Literal
import time

# 3 party imports
import numpy as np
import pandas as pd

# projekt imports
from xai_bench.base import BaseAttack, BaseDataset, BaseModel, BaseExplainer
from xai_bench.metrics.base_metric import BaseMetric


class MonteCarloAttack(BaseAttack):
    """
        The MonteCarloAttack creates candidates around the original data point. 
        All valid corrupted points are tested for their explanation distance to the
        input instance and the one with the best score is the result.

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

            num_candidates : int, default=100
                Number of steps.

            max_distance : float, default=0.1
                Maximum distance used for numerical feature updates (scaled by feature range).

            seed : int | None, default=None
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
        num_candidates: int = 100,
        max_distance: float = 0.1,
        num_samples_explainer: float = 100,
        seed: int = None,
        task: Literal["classification", "regression"] = "classification",
    ):
        super().__init__(model=model, task=task, epsilon=epsilon, stats=[self, "MonteCarloAttack"])
        self.dataset = dataset
        self.explainer = explainer
        self.metric = metric
        
        self.num_candidates = num_candidates
        self.max_distance = max_distance
        self.num_samples_explainer = num_samples_explainer
        
        self.protected_features = self.dataset.categorical_features
        self.cols = list(self.dataset.features.feature_names_model)
        self.col2idx = {c: i for i, c in enumerate(self.cols)}

        self.num_features = [f for f in (self.dataset.numerical_features or []) if f in self.col2idx]
        self.cat_features = [f for f in (self.dataset.categorical_features or []) if f in self.col2idx]

        self.ranges = getattr(self.dataset, "scaled_feature_ranges", None) or self.dataset.feature_ranges
        self.cat_vals = getattr(self.dataset, "scaled_categorical_values", None) or getattr(self.dataset, "categorical_values", {})

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)                         


    def _create_candidate(self, x: np.ndarray) -> np.ndarray:
        """
        Creates a candidate in proximity to the original point.

        Numerical features are perturbed by a random step within their
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
        x_candidate = x.copy()

        for feat in self.cols:
            idx = self.col2idx[feat]

            if feat in self.num_features:
                f_min, f_max = self.ranges[feat]
                span = f_max - f_min

                step = self.rng.uniform(
                    -self.max_distance * span,
                    self.max_distance * span
                )

                x_candidate[idx] = np.clip(
                    x_candidate[idx] + step, f_min, f_max
                )

            elif feat in self.cat_features:
                values = self.cat_vals.get(feat)
                if values is None or len(values) <= 1:
                    continue

                current_val = x_candidate[idx]
                possible_vals = [v for v in values if v != current_val]

                if possible_vals:
                    x_candidate[idx] = self.rng.choice(possible_vals)

        if self.is_attack_valid(x.reshape(1, -1), x_candidate.reshape(1, -1))[0]:
            return x_candidate

        return x


    def fit(self) -> None:
        """
            This needs to be def to make the base class happy :D 
        """
        pass 


    def _generate(self, x: np.ndarray) -> np.ndarray:
        """
            Generate an adversarial sample by creating candidates close to the original 
            instance and returning the best one according to the distance between the 
            explanation of the base and corrupted explanation.

            The method takes random steps from the origin `x` to create a pertubed sample 
            `x_adv` that fulfills the epsilon contraint.

            Parameters 
                x : np.ndarray
                    Original input sample of shape (d,).

            Returns
                np.ndarray
                    Adversarial sample found (shape (d,)).
        """  
        x_exp = self.explainer.explain(x.reshape(1, -1), self.num_samples_explainer)

        best_candidate = x
        best_score = -np.inf

        for _ in range(self.num_candidates):
            x_candidate = self._create_candidate(x)

            if np.allclose(x_candidate, x):
                continue

            cand_exp = self.explainer.explain(x_candidate.reshape(1, -1), self.num_samples_explainer)

            score = self.metric._compute(x_exp, cand_exp)

            if score > best_score:
                best_score = score
                best_candidate = x_candidate

        return best_candidate