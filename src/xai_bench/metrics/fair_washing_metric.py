# std-lib imports
from typing import Sequence, Optional, Union

# 3 pary imports
import numpy as np

# projekt imports 
from xai_bench.metrics.base_metric import BaseMetric


class FairnessWashingMetric(BaseMetric):
    """
        Weighted explanation-drift metric that can *downweight* (hide) or *upweight* (emphasize)
        changes on a chosen set of features.

        Typical use:
        - "wash" a sensitive/protected feature from evaluation by setting `mode="minimize"`
        (weight near 0 for those features), so changes there barely affect the score.
        - emphasize drift on specific features by setting `mode="maximize"` (high weight).

        The metric computes a weighted L1 distance between two explanation vectors:
            score = mean_i ( w_i * |e1_i - e2_i| )

        Parameters
        ----------
        focus:
            Features to focus on. Can be either:
            - indices (ints) into the explanation vector, or
            - feature names (strings) if you pass `feature_names`.
        feature_names:
            Optional list/array of feature names aligned with explanation vector order.
            Required if `focus` is given as strings.
        mode:
            - "maximize": strongly emphasize focused features (large weight).
            - "minimize": strongly de-emphasize focused features (small weight near 0).
        strength:
            Controls how strong the emphasis/de-emphasis is.
            - If mode="maximize": focused weight = 1 + strength
            - If mode="minimize": focused weight = max(0, 1 - strength)
            Recommended: 0.8..20 depending on how aggressive you want it.
        normalize_weights:
            If True, weights are normalized to have mean 1.0 (keeps score scale comparable).
        eps:
            Numerical stability constant.

        Notes
        -----
        - This metric is *not* a fairness metric. It is a reweighting trick that can be used
        to "wash out" or exaggerate drift on selected features.
        - Works with explanation arrays shaped (n_features,) in `_compute`, and (n, n_features)
        in `compute` inherited from `BaseMetric`.
    """

    def __init__(
        self,
        focus: Sequence[Union[int, str]],
        feature_names: Optional[Sequence[str]] = None,
        mode: str = "minimize",
        strength: float = 0.95,
        normalize_weights: bool = True,
        eps: float = 1e-12,
        name: str = "FairnessWashingMetric",
    ):
        super().__init__(name=name)
        self.feature_names = None if feature_names is None else list(feature_names)
        self.mode = str(mode).lower().strip()
        self.strength = float(strength)
        self.normalize_weights = bool(normalize_weights)
        self.eps = float(eps)

        if self.mode not in {"maximize", "minimize"}:
            raise ValueError("mode must be 'maximize' or 'minimize'")

        self.focus_idx = self._resolve_focus_indices(focus)


    def _resolve_focus_indices(self, focus: Sequence[Union[int, str]]) -> np.ndarray:
        if len(focus) == 0:
            return np.array([], dtype=int)
        
        if all(isinstance(x, (int, np.integer)) for x in focus):
            return np.array(list(focus), dtype=int)

        if all(isinstance(x, str) for x in focus):
            if self.feature_names is None:
                raise ValueError("feature_names must be provided when focus uses strings")
            name2idx = {n: i for i, n in enumerate(self.feature_names)}
            missing = [n for n in focus if n not in name2idx]
            if missing:
                raise ValueError(f"Unknown focus feature names: {missing}")
            return np.array([name2idx[n] for n in focus], dtype=int)

        raise TypeError("focus must be all ints or all strings")


    def _weights(self, d: int) -> np.ndarray:
        w = np.ones(d, dtype=float)

        if self.focus_idx.size > 0:
            if self.mode == "maximize":
                focused_w = 1.0 + self.strength
            else:  # minimize
                focused_w = max(0.0, 1.0 - self.strength)

            # clip invalid indices safely
            idx = self.focus_idx[(self.focus_idx >= 0) & (self.focus_idx < d)]
            w[idx] = focused_w

        if self.normalize_weights:
            w = w / (w.mean() + self.eps)

        return w


    def _compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        e1 = np.asarray(e1, dtype=float).reshape(-1)
        e2 = np.asarray(e2, dtype=float).reshape(-1)
        if e1.shape != e2.shape:
            raise ValueError(f"Shape mismatch: {e1.shape} vs {e2.shape}")

        w = self._weights(e1.size)
        return float(np.mean(w * np.abs(e1 - e2)))
