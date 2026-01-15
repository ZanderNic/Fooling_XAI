# std lib imports
from __future__ import annotations
from typing import Optional
import time

# 3-party imports 
import numpy as np
import lime.lime_tabular
import shap 

# projekt imports
from xai_bench.models.base_model import BaseModel
from xai_bench.explainer.base_explainer import BaseExplainer, Features, Explanation
from xai_bench.datasets.base_dataset import BaseDataset


class LimeTabularAdapter(BaseExplainer):
    """
        Adapter that exposes LIME (tabular) through the project's BaseExplainer interface.

        This implementation uses `lime.lime_tabular.LimeTabularExplainer` to create local,
        model-agnostic explanations for tabular data by fitting a weighted linear surrogate
        model around a single input point.

        Notes:
            - The returned feature weights are *local surrogate coefficients* (LIME)
            - Explanations are produced in the model input feature space (i.e., after any
              preprocessing such as scaling/one-hot encoding).
            - LIME is sampling-based; results can vary with random seeds and `num_samples`.
    """

    
    def __init__(
        self,
        dataset: BaseDataset,
        num_samples: int = 5000,
        num_features: Optional[int] = None,
        random_state: int = 42
        ):
        """
            Args:
                num_samples:
                    Number of perturbed samples used by LIME to fit the local surrogate.
                    Higher values improve stability but increase model query cost.

                num_features:
                    Maximum number of features returned by LIME. If None, uses all features.
                    Internally, LIME may still fit a full model but will return only the top-k
                    weights. This adapter converts the output into a dense vector.

                random_state:
                    Random seed forwarded to the LIME explainer to improve reproducibility.
        """
        super().__init__(stats=(self,"Lime Explainer"))
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.num_features = num_features 
        self.random_state = int(random_state)
        self._lime = None


    def fit(
        self, 
        reference_data: np.ndarray, 
        model: BaseModel, 
        features: Features
    ) -> None:
        """
            Initialize the underlying LIME explainer.

            Args:
                background:
                    Background data of shape (n_background, d) in the *model input space*.
                    LIME uses this to estimate feature distributions for perturbations.

                model:
                    Model wrapper/spec used by the benchmark. Must expose:
                    - `task` attribute: "classification" or "regression"
                    - `predict_scalar(X) -> (n,)` method

                features:
                    Feature specification. Must expose:
                    list[str] of length d
        """
        self.stats("fit")
        self.reference_data = reference_data
        self.model = model
        self.features = features
       
        mode = "regression" if model.task == "regression" else "classification"
        
        self._lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=reference_data,
            feature_names=features.feature_names_model,
            mode=mode,
            discretize_continuous=False,                # keep deterministic no binning
            random_state=self.random_state,
        )


    def explain(
        self, 
        X: np.ndarray,
        num_samples: int = None
    ) -> np.ndarray:
        """
            Compute a local LIME explanation for a single input sample.

            This method uses `LimeTabularExplainer` to generate a local, linear surrogate
            model around the given sample `x` and extracts the corresponding feature
            weights as a dense attribution vector.

            For classification tasks, the explanation is computed with respect to a
            specific target class. If `target` is None, LIME will internally select a
            class (typically the model's predicted class), and the corresponding
            explanation is returned.

            For regression tasks, `target` is ignored and the scalar model output is
            explained directly.

            The returned attribution vector contains one weight per input feature in
            the model feature space. Features not selected by LIME among the top-k
            features are assigned an attribution of zero.

            Args:
                x (np.ndarray):
                    One-dimensional input sample of shape (d,), expressed in the model
                    input feature space.

                target (Optional[int]):
                    Index of the class to explain for classification models. If None,
                    the class chosen internally by LIME is used.

            Returns:
                feature_importances (np.ndarray): 
                    Feature importances in the original feature order.

            Raises:
                AssertionError:
                    If the explainer has not been initialized via `fit()`.
        """
        assert self._lime is not None, "Call fit() first."
        self.stats("explain",X.shape[0] if X.ndim==2 else 1)
        model_prediction = self.model.predict(X)

        if self.model.task == "classification":
            exps = [self._lime.explain_instance(
                data_row=x,
                predict_fn=self.model.predict_proba,
                labels=self.dataset.classes,
                num_features=x.shape[0],
                num_samples= self.num_samples if num_samples is None else num_samples
            ) for x in X]
        else:  # regression
            exp = [self._lime.explain_instance(
                data_row=x,
                predict_fn=self.model.predict_scalar,
                num_features=x.shape[0],
                num_samples= self.num_samples if num_samples is None else num_samples
            ) for x in X]

        exp_values = []
        for i, exp in enumerate(exps):
            exp_values.append(list(dict(exp.as_list(model_prediction[i])).values()))
        exp_values = np.array(exp_values)

        # construct explanation object for compatibility with the dataset method
        explanation_object = Explanation(
            values=exp_values,
            feature_names=self.features.feature_names_model
        )

        return self.dataset.explanation_to_array(explanation_object)