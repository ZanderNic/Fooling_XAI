# std lib imports
from __future__ import annotations
from typing import Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import os

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
            def model_pred(X): # type: ignore
                prediction_probs = model.predict_proba(np.asarray(X, dtype=float))
                return prediction_probs.max(axis=1)
        else:  # regression
            def model_pred(X):
                return np.asarray(model.predict_scalar(np.asarray(X, dtype=float)), dtype=float)

        self._explainer = shap.KernelExplainer(model_pred, self._background)


    def explain(
        self, 
        X: np.ndarray,
        num_samples: Optional[int] = None
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
        assert self._explainer is not None, "Has to have explainer"
        shap_values = self._explainer.shap_values(
            X,
            nsamples=self.num_samples if num_samples is None else num_samples,
            random_state=self.random_state,
            silent=True
        )
        shap_values = np.asarray(shap_values)

        # construct explanation object for compatibility with the dataset method
        assert self.features is not None, "Has to have features"
        explanation_object = Explanation(
            values=shap_values,
            feature_names=self.features.feature_names_model
        )

        return self.dataset.explanation_to_array(explanation_object)


    def _create_thread_local_explainer(self):
        """
        Creates a function that returns a thread-local SHAP KernelExplainer
        instance as a duplicate of this explainer.

        Returns:
            get_explainer (Callable):
                A function that returns a thread-local SHAP KernelExplainer instance.
        """
        # Each thread will store its own KernelExplainer to avoid shared-state issues
        local = threading.local()
        def get_explainer():
            if getattr(local, 'explainer', None) is None:
                if self._background is None or self.model is None:
                    raise RuntimeError('ShapAdapter must be fitted before creating thread-local explainers')
                def model_pred(X):
                    assert self.model is not None," Has to have model"
                    if self.model.task == 'classification':
                        probs = self.model.predict_proba(np.asarray(X, dtype=float))
                        return probs.max(axis=1)
                    else:
                        return np.asarray(self.model.predict_scalar(np.asarray(X, dtype=float)), dtype=float)
                local.explainer = shap.KernelExplainer(model_pred, self._background)
            return local.explainer

        return get_explainer

    def explain_parallel(
        self,
        X: np.ndarray,
        num_samples: Optional[int] = None,
        n_workers: Optional[int] = None,
        batch_size: Optional[int] = 32
    ) -> np.ndarray:
        """
        Computes SHAP explanations in parallel using multiple threads with
        individual but identical SHAP KernelExplainer instances.

        Args:
            X (np.ndarray):
                Input samples of shape (n, d) in the model's expected input
                format.
    
            num_samples (int, optional):
                Number of samples to use for SHAP value estimation. If None,
                uses the default stored in this ShapAdapter.
            
            n_workers (int, optional):
                Number of parallel worker threads to use. If None, uses the
                number of CPU cores available.
    
            batch_size (int, optional):
                Number of samples to process in each batch per thread.
    
        Returns:
            feature_importances (np.ndarray):
                Feature importances for the given X in the original feature order.
        """
        assert self.model is not None, "Model not fitted"

        self.stats("explain",X)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = X.shape[0]

        if n_workers is None:
            n_workers = max(1, os.cpu_count() or 1)

        # create batches to reduce per-call overhead
        if not isinstance(batch_size, int) or batch_size <= 0:
            batch_size = 32
        batches = [X[i:i+batch_size] for i in range(0, n, batch_size)]

        get_explainer = self._create_thread_local_explainer()

        def _explain_batch(batch: np.ndarray) -> np.ndarray:
            expl = get_explainer()
            # call the thread-local KernelExplainer directly to avoid adapter-level shared state
            shap_vals = expl.shap_values(
                batch,
                nsamples=self.num_samples if num_samples is None else num_samples,
                random_state=self.random_state,
                silent=True
            )
            shap_vals = np.asarray(shap_vals)
            # use dataset helper to convert to final array format
            assert self.features is not None, "Has to have fefatues"
            explanation_obj = Explanation(
                values=shap_vals,
                feature_names=self.features.feature_names_model
            )
            return self.dataset.explanation_to_array(explanation_obj)

        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for res in ex.map(_explain_batch, batches):
                results.append(res)

        # stack batches back together
        results = [r if r.ndim == 2 else r.reshape(1, -1) for r in results]
        return np.vstack(results)
