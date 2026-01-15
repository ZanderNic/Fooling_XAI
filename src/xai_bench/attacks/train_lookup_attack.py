# std lib imports 
from typing import Literal
import time

# 3 party imports
import numpy as np
import pandas as pd

# projekt imports
from xai_bench.base import BaseAttack, BaseDataset, BaseModel, BaseExplainer
from xai_bench.metrics.base_metric import BaseMetric


class TrainLookupAttack(BaseAttack):
    """
        The TrainLookupAttack uses training samples with similar prediction values as 
        candidates and tests them. The candidate with the highest explanation distance 
        compared to the base instance is used.

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
        max_candidates: int = 100,
        num_samples_explainer: float = 100,
        seed: int = None,
        task: Literal["classification", "regression"] = "classification",
    ):
        super().__init__(model=model, task=task, epsilon=epsilon, stats=[self, "TrainLookupAttack"])
        self.dataset = dataset
        self.explainer = explainer
        self.metric = metric
        
        self.max_candidates = max_candidates
        self.num_samples_explainer = num_samples_explainer

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.X_train = None
        self.train_preds = None                        

    def fit(self) -> None:
            """
            Precompute predictions for the training data.
            """
            self.X_train = self.dataset.X_train

            if self.task == "classification":
                self.train_preds = self.model.predict_proba(self.X_train)
            else:
                self.train_preds = self.model.predict_scalar(self.X_train)


    def _generate(self, x: np.ndarray) -> np.ndarray:
        """
            Generate an adversarial sample using the train data.

            The method takes train examples with similar predictions and uses them as 
            candidates. The candidate with the highest explanation distance is returned.

            Parameters 
                x : np.ndarray
                    Original input sample of shape (d,).

            Returns
                np.ndarray
                    Adversarial sample found (shape (d,)).
        """        
        x_2d = x.reshape(1, -1)

        if self.task == "classification":
            pred_x = self.model.predict_proba(x_2d)
            pred_x_flat = pred_x.ravel()
            train_preds_flat = self.train_preds.reshape(len(self.train_preds), -1)
            pred_distances = np.sum(np.abs(train_preds_flat - pred_x_flat), axis=1)
        else:
            pred_x_val = self.model.predict_scalar(x_2d).ravel()
            train_preds_val = self.train_preds.ravel()
            pred_distances = np.abs(train_preds_val - pred_x_val)

        sorted_idx = np.argsort(pred_distances)
        candidate_idx = sorted_idx[: self.max_candidates]
        candidates = self.X_train.iloc[candidate_idx]

        valid_mask, _ = self.is_attack_valid(
            X=np.repeat(x_2d, len(candidates), axis=0),
            X_adv=candidates,
        )

        candidates = candidates[valid_mask]
        if len(candidates) == 0:
            return x

        exp_x = self.explainer.explain(x_2d, self.num_samples_explainer)
        
        candidates_array = candidates.to_numpy()

        exp_candidates = np.array([
            self.explainer.explain(candidates_array[i].reshape(1, -1), self.num_samples_explainer)
            for i in range(len(candidates_array))
        ])

        scores = self.metric.compute(np.repeat(exp_x, len(candidates), axis=0), exp_candidates)

        best_idx = np.argmax(scores)
        best_candidate = candidates_array[best_idx]
        return best_candidate