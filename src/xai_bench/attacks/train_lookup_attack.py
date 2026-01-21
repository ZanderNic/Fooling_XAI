# std lib imports 
from typing import Literal, Optional

# 3 party imports
import numpy as np
import pandas as pd

# projekt imports
from xai_bench.base import BaseAttack, BaseDataset, BaseModel, BaseExplainer
from xai_bench.metrics.base_metric import BaseMetric


class TrainLookupAttack(BaseAttack):
    """
        The TrainLookupAttack uses test samples with similar prediction values as 
        candidates and tests them. The candidate with the highest explanation distance 
        compared to the base instance is used. Only close candidates are considered.

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

            max_candidates: int, default=100
                Maximum number of candidates.

            max_distance: int, default=100
                Maximum distance of a numerical feature.

            max_cat_ratio: int, default=100
                Maximum ratio of changed categorical features (minimum 1 allowed).

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
        max_candidates: int = 100,
        max_distance: float = 0.3,
        max_cat_ratio: float = 0.3,
        num_samples_explainer: int = 100,
        seed: Optional[int] = None,
        task: Literal["classification", "regression"] = "classification",
    ):
        super().__init__(model=model, task=task, epsilon=epsilon, stats=[self, "TrainLookupAttack"],dataset=dataset)
        self.explainer = explainer
        self.metric = metric
        
        self.max_candidates = max_candidates
        self.max_distance = max_distance
        self.max_cat_ratio = max_cat_ratio
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

        self.X_train = None
        self.train_preds = None                        


    def fit(self) -> None:
            """
            Precompute predictions for the training data.
            """
            self.X_train = self.dataset.X_test_scaled

            if self.task == "classification":
                self.train_preds = self.model.predict_proba(self.X_train)
            else:
                self.train_preds = self.model.predict_scalar(self.X_train)


    def _ensure_proximity(self, x: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        Check proximity constraints for numerical and categorical features.

        Numerical:
            |x_i - x'_i| <= max_distance for all numerical features

        Categorical:
            At most max_cat_ratio of categorical features may differ,
            but at least 1 categorical feature is allowed to differ.
        """
        num_idx = [self.col2idx[f] for f in self.numerical_features]
        cat_idx_groups = [[self.col2idx[c] for c in self.feature_mapping[f]] for f in self.categorical_features]

        num_diff = np.abs(candidates[:, num_idx] - x[num_idx])
        num_okay = np.all(num_diff <= self.max_distance, axis=1)

        cat_okay = []
        for row in candidates:
            n_diff = 0
            for idxs in cat_idx_groups:
                active = np.where(row[idxs] == 1)[0]
                original = np.where(x[idxs] == 1)[0]
                if len(active) != 1 or len(original) != 1:
                    continue
                if active[0] != original[0]:
                    n_diff += 1
            cat_okay.append(n_diff <= max(1, np.ceil(self.max_cat_ratio * len(self.categorical_features))))

        cat_okay = np.array(cat_okay)
        return num_okay & cat_okay


    def _generate(self, x: np.ndarray) -> np.ndarray:
        """
        Generate an adversarial sample using the test data.

        The method takes test examples with similar predictions and uses them as 
        candidates. The candidate with the highest explanation distance is returned.
        Only instances close to the x are considered.
        """
        x_2d = x.reshape(1, -1)
        assert self.X_train is not None
        assert self.train_preds is not None

        valid_mask, _ = self.is_attack_valid(
            X=np.repeat(x_2d, len(self.X_train), axis=0),
            X_adv=self.X_train.values,
        )
        if not np.any(valid_mask):
            return x

        candidates = self.X_train.values[valid_mask]

        proximity_mask = self._ensure_proximity(x, candidates)
        if not np.any(proximity_mask):
            return x

        candidates = candidates[proximity_mask]

        if len(candidates) > self.max_candidates:
            indices = self.rng.choice(len(candidates), size=self.max_candidates, replace=False)
            candidates = candidates[indices]

        exp_x = self.explainer.explain(x_2d, self.num_samples_explainer)
        exp_candidates = self.explainer.explain_parallel(
            pd.DataFrame(candidates, columns=self.X_train.columns),
            num_samples=self.num_samples_explainer,
            n_workers=4
        )

        scores = self.metric.compute(np.repeat(exp_x, len(exp_candidates), axis=0), exp_candidates)
        best_idx = np.argmax(scores)
        best_candidate = candidates[best_idx]

        return best_candidate