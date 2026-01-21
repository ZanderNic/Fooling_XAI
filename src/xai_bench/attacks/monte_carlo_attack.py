# std lib imports 
from typing import Literal, Optional

# 3 party imports
import numpy as np

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
        num_candidates: int = 100,
        max_distance: float = 0.1,
        p_numerical: float = 0.8,
        p_categorical: float = 0.1,
        num_samples_explainer: int = 100,
        seed: Optional[int] = None,
        task: Literal["classification", "regression"] = "classification",
    ):
        super().__init__(model=model, task=task, epsilon=epsilon, stats=[self, "MonteCarloAttack"],dataset=dataset)
        self.explainer = explainer
        self.metric = metric
        
        self.num_candidates = num_candidates
        self.max_distance = max_distance
        self.p_numerical = p_numerical
        self.p_categorical = p_categorical
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

        for feat in self.numerical_features:
            if self.rng.random() > self.p_numerical:
                continue

            idx = self.col2idx[feat]

            f_min, f_max = self.ranges[feat]
            span = f_max - f_min

            step = self.rng.uniform(
                -self.max_distance * span,
                self.max_distance * span
            )

            x_candidate[idx] = np.clip(
                x_candidate[idx] + step, f_min, f_max
            )

        for feat in self.categorical_features:
            if self.rng.random() > self.p_categorical:
                continue

            cols = self.feature_mapping[feat]

            idxs = [self.col2idx[c] for c in cols]

            active = [i for i in idxs if x_candidate[i] == 1]

            if len(active) != 1:
                continue

            current_idx = active[0]
            other_idxs = [i for i in idxs if i != current_idx]

            if not other_idxs:
                continue

            new_idx = self.rng.choice(other_idxs)

            x_candidate[current_idx] = 0.0
            x_candidate[new_idx] = 1.0

        if self.is_attack_valid(x.reshape(1, -1), x_candidate.reshape(1, -1))[0]:
            return x_candidate

        return x


    def fit(self) -> None:
        """
            Necessary for interface
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

            cand_exp = self.explainer.explain(x_candidate.reshape(1, -1), self.num_samples_explainer)

            score = self.metric.compute(x_exp, cand_exp)

            if score > best_score:
                best_score = score
                best_candidate = x_candidate

        return best_candidate