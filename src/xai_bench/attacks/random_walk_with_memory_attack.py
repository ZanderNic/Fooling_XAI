# std lib imports 
from typing import Literal, Optional

# 3 party imports
import numpy as np

# projekt imports
from xai_bench.base import BaseAttack


class RandomWalkWithMemoryAttack(BaseAttack):
    """
        The RandomWalkWithMemoryAttack additionally uses the explainer and metric to compute the 
        explanation distance for each step. The best scoring instance is used as attack. 
        Multiple runs are executed to find a good pertubation. 

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

            num_runs : int, default=10
                Number of runs.

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
        dataset,
        model,
        explainer,
        metric,
        epsilon: float = 0.05,
        num_runs: int = 10,
        num_steps: int = 100,
        step_len: float = 0.1,
        num_samples_explainer: int = 100,
        seed: Optional[int] = None,
        task: Literal["classification", "regression"] = "classification",
    ):
        super().__init__(model=model, task=task, epsilon=epsilon, stats=[self, "RandomWalkAttack"], dataset=dataset)
        self.explainer = explainer
        self.metric = metric
        
        self.num_runs = num_runs
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
            Generate an adversarial sample using multiple random walks and memory to save the 
            global best occuring instance that has the highest explanation difference.

            The method takes random steps from the origin `x` to create a pertubed sample 
            `x_adv` that fulfills the epsilon contraint. All explanation distances of the steps 
            are compared to the global best attack. This is done for multiple runs and the 
            final best instance is returned.

            Parameters 
                x : np.ndarray
                    Original input sample of shape (d,).

            Returns
                np.ndarray
                    Adversarial sample found (shape (d,)).
        """       
        x_exp = self.explainer.explain(x.reshape(1, -1), self.num_samples_explainer)

        best_global_x = x.copy()
        best_global_score = -np.inf

        for _ in range(self.num_runs):
            current_x = x.copy()

            for _ in range(self.num_steps):
                current_x = self._step(current_x)
                current_exp = self.explainer.explain(current_x.reshape(1, -1), self.num_samples_explainer)
                score = self.metric.compute(x_exp, current_exp)[0]

                if score > best_global_score:
                    best_global_score = score
                    best_global_x = current_x.copy()

        return best_global_x