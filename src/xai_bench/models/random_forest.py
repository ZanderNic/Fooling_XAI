from __future__ import annotations
from typing import Optional, Literal, TypedDict, Union


# 3 party imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# projekt imports
from xai_bench.models.base_model import BaseModel


# ----------------------------
# Helpers
# ----------------------------

def _to_numpy(X) -> np.ndarray:
    """Accepts numpy array / torch tensor / list-like and returns float32 numpy."""
    if isinstance(X, np.ndarray):
        arr = X
    else:
        arr = np.asarray(X)
    if arr.dtype == object:
        raise TypeError("X must be numeric, got dtype=object.")
    return arr.astype(np.float32, copy=False)

def _check_1d_vector(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1D of shape (n,).")
    return y

class rf_kwargs(TypedDict):
    n_estimators: int
    random_state: int
    n_jobs: int
    max_depth: Optional[int]

###################################################



class SKRandomForest(BaseModel):
    """
    Wrapper around sklearn RandomForestClassifier/Regressor.

    - classification: implements predict_proba
    - regression: implements predict_scalar
    """

    def __init__(
        self,
        task: Literal["classification","regression"],
        *,
        n_estimators: int = 300,
        random_state: int = 0,
        n_jobs: int = -1,
        max_depth: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(task,stats=(self,"RF"))
        self._rf_kwargs = rf_kwargs(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            max_depth=max_depth,
            **kwargs,
        )
        self.model = None

    def fit(self, X, y) -> "SKRandomForest":
        self.stats("fit",X)
        Xn = _to_numpy(X)
        yn = _check_1d_vector(y)
        
        self.model = RandomForestClassifier(**self._rf_kwargs)
        self.model.fit(Xn, yn)
        
        return self

    def predict_proba(self, X) -> np.ndarray:
        self.stats("predict_proba",X)
        if self.task != "classification":
            raise NotImplementedError("predict_proba is only defined for classification models.")
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        proba = self.model.predict_proba(_to_numpy(X))
        proba = np.asarray(proba, dtype=np.float64)
        
        if proba.ndim != 2:
            raise ValueError(f"predict_proba must return array of shape (n, C) but is {proba.shape}")
        
        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("predict_proba rows must sum to 1.")
        
        return proba

    def predict_scalar(self, X) -> np.ndarray:
        self.stats("predict_scalar",X)
        if self.task != "regression":
            raise NotImplementedError("predict_scalar is only defined for regression models.")
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        
        pred = self.model.predict(_to_numpy(X))
        pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim != 1:
            raise ValueError("predict_scalar must return 1D array of shape (n,)")
        return pred