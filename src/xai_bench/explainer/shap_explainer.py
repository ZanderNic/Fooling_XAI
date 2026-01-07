# std lib imports
from __future__ import annotations
from typing import Optional
import time

# 3-party imports 
import numpy as np
import shap

# projekt imports
from xai_bench.explainer.base_explainer import BaseExplainer

class ShapAdapter(BaseExplainer):
    def __init__(self, nsamples: int = 2000, background_size: int = 200, random_state: int = 42):
        self.nsamples = int(nsamples)
        self.background_size = int(background_size)
        self.random_state = int(random_state)

        self.model = None
        self.features = None
        self._background = None

    def fit(
        self, 
        reference_data: np.ndarray, 
        model, 
        features,
        use_all_data: bool = False
    ) -> None:
        self.model = model
        self.features = features

        X = np.asarray(reference_data, dtype=float)
        
        if not use_all_data:
            if self.background_size > 0 and X.shape[0] > self.background_size:
                rng = np.random.default_rng(self.random_state)
                X = X[rng.choice(X.shape[0], size=self.background_size, replace=False)]
            
        self._background = X


    def explain(
        self, 
        x: np.ndarray, 
        target: Optional[int] = None
    ) -> np.array:
        x = np.asarray(x, dtype=float).reshape(1, -1)

        if self.model.task == "classification":
            if target is None:
                raise ValueError("For classification, `target` must be provided.")

            def model_pred(X):
                return self.model.predict_proba(np.asarray(X, dtype=float))[:, int(target)]

            explainer = shap.KernelExplainer(model_pred, self._background)
            shap_values = explainer.shap_values(x, nsamples=self.nsamples, random_state=self.random_state, silent=True)
            return Explanation(
                attributions=np.asarray(shap_values, dtype=float).reshape(-1),
                base_value=float(explainer.expected_value),
                target=int(target)
            )
        else:
            def model_pred(X):
                return np.asarray(self.model.predict_scalar(np.asarray(X, dtype=float), target=None), dtype=float).reshape(-1)

            explainer = shap.KernelExplainer(model_pred, self._background)
            shap_values = explainer.shap_values(x, nsamples=self.nsamples, random_state=self.random_state, silent=True)
        
            return shap_values