# std lib imports
from __future__ import annotations
from typing import Optional
import time

# 3-party imports 
import numpy as np
import shap

# projekt imports
from xai_bench.explainer.base_explainer import BaseExplainer, Features, Explanation
from xai_bench.datasets.base_dataset import BaseDataset
from xai_bench.models.base_model import BaseModel


class ShapAdapter(BaseExplainer):
    def __init__(
        self,
        dataset: BaseDataset,
        num_samples: int = 2000,
        background_size: int = 200,
        random_state: int = 42
    ):
        super().__init__(stats=(self,"ShapExplainer counter"))
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.background_size = int(background_size)
        self.random_state = int(random_state)

        self.model: Optional[BaseModel] = None
        self.features: Optional[Features] = None
        self._background = None
        self._explainer: Optional[shap.KernelExplainer] = None

    def fit(
        self,
        reference_data: np.ndarray,
        model: BaseModel,
        features: Features,
        use_all_data: bool = False
    ) -> None:
        self.stats("fit")
        self.model = model
        self.features = features

        # prepare background data for SHAP
        X = np.asarray(reference_data, dtype=float)
        
        if not use_all_data:
            if self.background_size > 0 and X.shape[0] > self.background_size:
                rng = np.random.default_rng(self.random_state)
                X = X[rng.choice(X.shape[0], size=self.background_size, replace=False)]
            
        self._background = X

        # set up a shap expalainer with the background data and model prediction function
        if self.model.task == "classification":
            def model_pred(X):
                prediction_probs = model.predict_proba(np.asarray(X, dtype=float))
                return prediction_probs.max(axis=1)
        else:  # regression
            def model_pred(X):
                return np.asarray(model.predict_scalar(np.asarray(X, dtype=float)), dtype=float)

        self._explainer = shap.KernelExplainer(model_pred, self._background)


    def explain(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute a SHAP explanation and aggregate it according to the dataset's feature mapping.
        Works for classification and regression.

        Args:
            X (np.ndarray):
                Input samples of shape (n, d) in the model's expected input format.

        Returns:
            feature_importances (np.ndarray): 
                Feature importances in the original feature order.
        """
        assert self.model is not None, "Model not fitted"

        X = np.asarray(X, dtype=float)
        self.stats("explain",X)

        # produce explanations
        shap_values = self._explainer.shap_values(
            X,
            nsamples=self.num_samples,
            random_state=self.random_state,
            silent=True
        )
        shap_values = np.asarray(shap_values)

        # construct explanation object for compatibility with the dataset method
        explanation_object = Explanation(
            values=shap_values,
            feature_names=self.features.feature_names_model
        )

        return self.dataset.explanation_to_array(explanation_object)