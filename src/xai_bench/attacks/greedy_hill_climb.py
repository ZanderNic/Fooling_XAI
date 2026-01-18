# std lib imports 
from typing import Literal, Optional

# 3 party imports
import numpy as np

# projekt imports
from xai_bench.base import BaseAttack, BaseDataset, BaseModel, BaseExplainer
from xai_bench.metrics.base_metric import BaseMetric


class GreedyHillClimb(BaseAttack):
    """
        Greedy hill-climbing attack that searches for a perturbed input `x_adv` within an
        epsilon-ball around the original sample `x`, such that the explanation of the model
        output changes as much as possible (according to a user-defined explanation metric).

        The algorithm iteratively samples multiple candidate perturbations ("directions")
        around the current point and greedily moves to the candidate that maximizes the
        explanation difference relative to the original explanation.

        Supports mixed feature spaces:
        - Numerical features: small additive steps within observed feature ranges.
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
                explanation of a candidate adversarial sample. The hill-climbing procedure tries
                to maximize this metric. Typically, this is a distance or similarity measure
                computed on explanation vectors.

            epsilon : float, default=0.1
                Maximum allowed perturbation size (constraint handled by `is_attack_okay`).

            num_climbs : int, default=100
                Number of hill-climbing iterations per try.

            num_derections : int, default=100
                Number of candidate perturbations sampled per hill-climb step.

            max_trys : int, default=1
                Number of independent restarts of the hill-climb procedure.

            step_len : float, default=0.001
                Step length used for numerical feature updates (scaled by feature range).

            proba_numeric : float, default=0.7
                Probability that a sampled candidate move modifies a numerical feature.
                Otherwise a categorical feature is modified.

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
        epsilon: float = 0.1,
        num_climbs: int = 30,
        num_derections: int = 30,
        max_trys : int = 1,
        step_len: float = 0.0001,
        num_samples_explainer: float = 100,
        proba_numeric: float = 0.7,
        seed: Optional[int] = None,
        task: Literal["classification", "regression"] = "classification",
    ):
        super().__init__(model,task=task, epsilon=epsilon, stats=[self, "GreedyHillClimb"],dataset=dataset)
        self.model = model
        self.explainer = explainer
        self.metric = metric
        self.epsilon = epsilon
        
        self.num_climbs = num_climbs
        self.max_trys = max_trys
        self.num_derections = num_derections
        self.step_len = step_len
        self.proba_numeric = proba_numeric if proba_numeric > 0 and proba_numeric < 1 else 0.8
        self.num_samples_explainer = num_samples_explainer
        
        self.protected_features = self.dataset.categorical_features
        self.cols = list(self.dataset.features.feature_names_model)
        self.col2idx = {c: i for i, c in enumerate(self.cols)}

        self.num_features = [f for f in (self.dataset.numerical_features or []) if f in self.col2idx]
        self.cat_features = [f for f in (self.dataset.categorical_features or []) if f in self.col2idx]

        self.ranges = getattr(self.dataset, "scaled_feature_ranges", None) or self.dataset.feature_ranges
        self.cat_vals = getattr(self.dataset, "scaled_categorical_values", None) or getattr(self.dataset, "categorical_values", {})

        # for reproducability if seed is None np will take random seed 
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)                         


    def _sample_directions(
        self,
        x: np.ndarray,
        num_directions: int = None,
        p_sparse: float = 0.6,
        p_single: float = 0.3,
        p_escape: float = 0.1,
    ) -> list[np.ndarray]:
        """
            Sample candidate perturbations ("directions") around a given input sample using a
            probabilistic mixture of move types.

            The method draws candidate directions according to three heuristics that balance
            exploration and exploitation:

                - Sparse numeric directions (default 60%):
                    Modify 2-5 randomly chosen numerical features using a random (non-axis-aligned)
                    direction vector. This enables efficient local exploration without changing
                    too many dimensions at once.

                - Single-feature moves (default 30%):
                    Modify exactly one feature (numerical or categorical). This is useful for
                    fine-tuning near local optima.

                - Escape moves (default 10%):
                    Perform a larger jump to escape local plateaus. This is implemented as either:
                        1 categorical switch + 2 numerical updates, or
                        a dense numerical move affecting 6-10 features.

            The probabilities of these move types can be customized via `p_sparse`, `p_single`,
            and `p_escape`. They should sum approximately to 1.0.

            Parameters
            ----------
            x : np.ndarray
                Input sample of shape (d,) in model input space.

            num_directions : Optional[int], default=None
                Number of candidate perturbations to generate. If None, `self.num_derections`
                is used.

            p_sparse : float, default=0.6
                Probability of sampling a sparse numeric (non-axis-aligned) direction.

            p_single : float, default=0.3
                Probability of sampling a single-feature move.

            p_escape : float, default=0.1
                Probability of sampling an escape move.

            Returns
            -------
            list[np.ndarray]
                List of perturbed candidate samples, each of shape (d,).
        """
        num_directions = num_directions or self.num_derections
        moves: list[np.ndarray] = []

        has_num = len(self.num_features) > 0
        has_cat = len(self.cat_features) > 0

        psum = p_sparse + p_single + p_escape
        p_sparse, p_single, p_escape = (p_sparse / psum, p_single / psum, p_escape / psum)      # normalize probabilities

        for _ in range(num_directions):
            x_new = x.copy()
            r = self.rng.random()

            # Sparse numeric random direction
            if r < p_sparse and has_num:
                k = int(self.rng.integers(2, 6)) 
                k = min(k, len(self.num_features))
                feats = self.rng.choice(self.num_features, size=k, replace=False)

                d = self.rng.normal(size=k)
                d /= (np.linalg.norm(d) + 1e-12)

                for j, f in enumerate(feats):
                    idx = self.col2idx[f]
                    low, high = self.ranges[f]
                    step = self.step_len * (high - low) * d[j]
                    x_new[idx] = np.clip(x_new[idx] + step, low, high)

            # Single-feature move 
            elif r < p_sparse + p_single:
                do_num = has_num and ((not has_cat) or (self.rng.random() < self.proba_numeric))
                if do_num:
                    f = self.rng.choice(self.num_features)
                    idx = self.col2idx[f]
                    low, high = self.ranges[f]
                    step = self.step_len * (high - low) * self.rng.choice([-1.0, 1.0])
                    x_new[idx] = np.clip(x_new[idx] + step, low, high)
                elif has_cat:
                    f = self.rng.choice(self.cat_features)
                    idx = self.col2idx[f]
                    vals = self.cat_vals.get(f, None)
                    if vals is not None and len(vals) > 1:
                        cur = x_new[idx]
                        cand = vals[vals != cur]
                        if len(cand) > 0:
                            x_new[idx] = self.rng.choice(cand)

            # Escape move (larger jump)
            else:
                if has_num:
                    if has_cat and len(self.num_features) >= 2 and (self.rng.random() < 0.6):
                        f = self.rng.choice(self.cat_features)
                        idx = self.col2idx[f]
                        vals = self.cat_vals.get(f, None)
                        if vals is not None and len(vals) > 1:
                            cur = x_new[idx]
                            cand = vals[vals != cur]
                            if len(cand) > 0:
                                x_new[idx] = self.rng.choice(cand)

                        feats = self.rng.choice(self.num_features, size=2, replace=False)
                        d = self.rng.normal(size=2)
                        d /= (np.linalg.norm(d) + 1e-12)

                        for j, f2 in enumerate(feats):
                            idx2 = self.col2idx[f2]
                            low, high = self.ranges[f2]
                            step = self.step_len * (high - low) * d[j]
                            x_new[idx2] = np.clip(x_new[idx2] + step, low, high)
                    else:
                        k = int(self.rng.integers(6, 11)) 
                        k = min(k, len(self.num_features))
                        if k > 0:
                            feats = self.rng.choice(self.num_features, size=k, replace=False)
                            d = self.rng.normal(size=k)
                            d /= (np.linalg.norm(d) + 1e-12)

                            for j, f in enumerate(feats):
                                idx = self.col2idx[f]
                                low, high = self.ranges[f]
                                step = self.step_len * (high - low) * d[j]
                                x_new[idx] = np.clip(x_new[idx] + step, low, high)

                elif has_cat:
                    f = self.rng.choice(self.cat_features)
                    idx = self.col2idx[f]
                    vals = self.cat_vals.get(f, None)
                    if vals is not None and len(vals) > 1:
                        cur = x_new[idx]
                        cand = vals[vals != cur]
                        if len(cand) > 0:
                            x_new[idx] = self.rng.choice(cand)

            moves.append(x_new)

        return moves


    def fit(self) -> None:
        """
            This needs to be def to make the base class happy :D 
        """
        pass 


    def _generate(self, x: np.ndarray) -> np.ndarray:
        """
            Generate an adversarial sample using greedy hill climbing.

            The method computes the original explanation for `x` once and then searches for a
            perturbed sample `x_adv` that maximizes the explanation distance to the original
            explanation while staying within the allowed epsilon constraint.

            Workflow:
                1) Compute explanation of the original sample `x`.
                2) Initialize `current_x = x`.
                3) For `num_climbs` steps:
                    - sample `num_derections` candidate perturbations around `current_x`
                    - compute explanations for candidates
                    - evaluate explanation difference metric
                    - greedily move to the best candidate
                4) Track and return the best candidate found across all tries.

            Parameters 
                x : np.ndarray
                    Original input sample of shape (d,).

            Returns
                np.ndarray
                    Best adversarial sample found (shape (d,)).
        """
        # t0 = time.perf_counter()
        
        x_exp = self.explainer.explain(x.reshape(1, -1))
                
        best_global_x = x.copy()                                        # here the x with the best attack metric will be saved   
        best_global_metric = 0                                          # here the difference in the metric betwean the real x and the best attack x will be saved 

        for _ in range(self.max_trys):
            current_x = x.copy() 
            
            for _ in range(self.num_climbs):
                best_local = current_x.copy()                                       # to ensure we make a step we will save this here                                             
                best_local_metric = 0
                
                # search in self.num_directions different directions
                candidates = self._sample_directions(current_x, num_directions = self.num_derections)

                for canidate in candidates:
                    canidate_exp = self.explainer.explain(canidate.reshape(1, -1), self.num_samples_explainer)
                    metric = self.metric.compute(canidate_exp, x_exp)
                    
                    if metric > best_local_metric and (self.is_attack_valid(current_x.reshape(1, -1), x.reshape(1, -1), self.epsilon)[0]):  # check if current_x is still valid 
                        best_local = canidate
                        best_local_metric = metric
                        current_x = canidate
                
                if best_local_metric > best_global_metric:
                    best_global_metric = best_local_metric
                    best_global_x = best_local   

        # t1 = time.perf_counter()
        # adv_exp = self.explainer.explain(best_global_x.reshape(1, -1))
        # print("\n--- GreedyHillClimb Log ---")
        # print(f"total _generate   : {(t1 - t0):.4f}s")
        # print(f"best metric value : {float(np.asarray(metric).mean()):.6f}")
        # print("x_exp   :", x_exp)
        # print("adv_exp :", adv_exp)
        # print("--- end debug ---\n")
                  
        return best_global_x
    
    
    def _trace_generate(
        self,
        x: np.ndarray,
        max_steps: Optional[int] = None,
        num_directions: Optional[int] = None,
        keep_top_k: int = 15,
    ) -> tuple[np.ndarray, dict]:
        """
            Generate an adversarial sample like `_generate`, but also return a detailed trace of
            the greedy hill-climb search for visualization/debugging.

            This is intended for **notebook demos** and **debugging** (it stores candidates and
            scores per step and can become large).

            Parameters
            ----------
            x:
                Single input sample of shape (d,).
            max_steps:
                Number of hill-climb iterations to run. If None, uses `self.num_climbs`.
            num_directions:
                Number of candidate perturbations per step. If None, uses `self.num_derections`.
            keep_top_k:
                Store only the top-k candidates (by score) per step to reduce memory and plotting
                clutter. If <= 0 or None, stores all candidates.

            Returns
            -------
            x_adv:
                Best adversarial sample found (shape (d,)).
            trace:
                Dict with keys:
                - "x0": original sample (d,)
                - "path": array of visited points, shape (t+1, d)
                - "best_scores": best score per step, shape (t,)
                - "picked_idx": index of picked candidate within the stored candidate subset, shape (t,)
                - "cand_x": list of arrays (m_t, d) stored candidates per step (top-k or full)
                - "cand_scores": list of arrays (m_t,) stored scores per step (aligned with cand_x)
                - "explainer_time": total seconds spent in explainer calls (float)
                - "total_time": total seconds for trace_generate (float)

            Notes
            -----
            - Uses **batch explanation** for candidates (`explainer.explain(C)`) which is usually
            much faster than explaining each candidate individually.
            - Uses `self.is_attack_valid` (2D inputs) to enforce the epsilon constraint.
        """
        import time

        print("[WARNING] THIS FUNKTION IS ONLY FOR VISUALISATION")


        t0 = time.perf_counter()
        max_steps = self.num_climbs if max_steps is None else int(max_steps)
        num_directions = self.num_derections if num_directions is None else int(num_directions)

        x0 = np.asarray(x, dtype=float).reshape(-1)
        current_x = x0.copy()

        exp_time = 0.0
        t_exp0 = time.perf_counter()
        x_exp = self.explainer.explain(x0.reshape(1, -1))
        exp_time += time.perf_counter() - t_exp0

        path = [current_x.copy()]
        best_scores = []
        picked_idx = []
        cand_x_hist = []
        cand_s_hist = []

        for _ in range(max_steps):
            if not self.is_attack_valid(current_x.reshape(1, -1), x0.reshape(1, -1), self.epsilon)[0]:
                break

            candidates = self._sample_directions(current_x, num_directions=num_directions)
            C = np.asarray(candidates, dtype=float)  # (m, d)

            t_exp0 = time.perf_counter()
            E_c = self.explainer.explain(C)
            exp_time += time.perf_counter() - t_exp0

            # metric.compute may return (m,) or (m,1) depending on implementation
            s = self.metric.compute(E_c, np.repeat(x_exp, repeats=len(C), axis=0))
            s = np.asarray(s).reshape(-1)

            j_full = int(np.argmax(s))
            best_scores.append(float(s[j_full]))

            if keep_top_k is not None and keep_top_k > 0 and len(s) > keep_top_k:
                top_idx = np.argsort(s)[-keep_top_k:]
                C_store = C[top_idx]
                s_store = s[top_idx]
                cand_x_hist.append(C_store)
                cand_s_hist.append(s_store)

                # map chosen index into stored subset
                j_store = int(np.where(top_idx == j_full)[0][0]) if j_full in top_idx else int(np.argmax(s_store))
                picked_idx.append(j_store)
            else:
                cand_x_hist.append(C)
                cand_s_hist.append(s)
                picked_idx.append(j_full)

            # greedy step
            current_x = C[j_full].copy()
            path.append(current_x.copy())

        t1 = time.perf_counter()
        x_adv = path[-1].copy()

        trace = {
            "x0": x0,
            "path": np.asarray(path),
            "best_scores": np.asarray(best_scores, dtype=float),
            "picked_idx": np.asarray(picked_idx, dtype=int),
            "cand_x": cand_x_hist,
            "cand_scores": cand_s_hist,
            "explainer_time": float(exp_time),
            "total_time": float(t1 - t0),
        }
        return x_adv, trace