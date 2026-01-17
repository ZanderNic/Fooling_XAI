# std lib imports
from __future__ import annotations
from typing import Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import os

# 3-party imports 
import numpy as np
import lime.lime_tabular

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
        self.reference_data: Optional[np.ndarray] = None
        self.model: Optional[BaseModel] = None
        self.features: Optional[Features] = None


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

        assert self.model.task in ["classification","regression"], "LIME only supports classification and regression tasks"
        mode = model.task
        
        self._lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.reference_data,
            feature_names=self.features.feature_names_model,
            mode=mode,
            discretize_continuous=False,  # keep deterministic no binning
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
            X (np.ndarray):
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
        assert 1 <= X.ndim <= 2, "Input X must be 1D or 2D array."
        self.stats("explain", X.shape[0] if X.ndim==2 else 1)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
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
            exps = [self._lime.explain_instance(
                data_row=x,
                predict_fn=self.model.predict_scalar,
                num_features=x.shape[0],
                num_samples= self.num_samples if num_samples is None else num_samples
            ) for x in X]

        exp_values = []
        for i, exp in enumerate(exps):
            # model_prediction is ignored by LIME in case of regression task
            exp_values.append(list(dict(exp.as_list(model_prediction[i])).values()))
        exp_values = np.array(exp_values)

        # construct explanation object for compatibility with the dataset method
        explanation_object = Explanation(
            values=exp_values,
            feature_names=self.features.feature_names_model
        )

        return self.dataset.explanation_to_array(explanation_object)


    def _create_thread_local_explainer(self):
        """
        Creates a function that returns a thread-local LIME TabularExplainer
        instance as a duplicate of this explainer.

        Returns:
            get_explainer (Callable):
                A function that returns a thread-local LIME TabularExplainer instance.
        """
        # Each thread will store its own KernelExplainer to avoid shared-state issues
        local = threading.local()

        def get_explainer():
            if getattr(local, 'explainer', None) is None:
                if self.reference_data is None or self.model is None or self.features is None:
                    raise RuntimeError(
                        'LimeTabularAdapter must be fitted before creating thread-local explainers'
                    )
                
                local.explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=self.reference_data,
                    feature_names=self.features.feature_names_model,
                    mode=self.model.task,
                    discretize_continuous=False,
                    random_state=self.random_state,
                )
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
        Computes LIME explanations in parallel using multiple threads with
        individual but identical LIME TabularExplainer instances.

        Args:
            X (np.ndarray):
                Input samples of shape (n, d) in the model's expected input
                format.

            num_samples (int, optional):
                Number of samples to use for LIME explanation. If None, uses
                the default stored in this LimeTabularAdapter.
            
            n_workers (int, optional):
                Number of parallel worker threads to use. If None, uses the
                number of CPU cores available.
    
            batch_size (int, optional):
                Number of samples to process in each batch per thread.
    
        Returns:
            feature_importances (np.ndarray):
                Feature importances for the given X in the original feature order.
        """
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
            model_prediction = self.model.predict(X)
            expl = get_explainer()

            # call the thread-local LIME explainer directly to avoid adapter-level shared state
            if self.model.task == "classification":
                exps = [expl.explain_instance(
                    data_row=x,
                    predict_fn=self.model.predict_proba,
                    labels=self.dataset.classes,
                    num_features=x.shape[0],
                    num_samples= self.num_samples if num_samples is None else num_samples
                ) for x in batch]
            else:  # regression
                exps = [expl.explain_instance(
                    data_row=x,
                    predict_fn=self.model.predict_scalar,
                    num_features=x.shape[0],
                    num_samples= self.num_samples if num_samples is None else num_samples
                ) for x in batch]

            exp_values = []
            for i, exp in enumerate(exps):
                # model_prediction is ignored by LIME in case of regression task
                exp_values.append(list(dict(exp.as_list(model_prediction[i])).values()))
            exp_values = np.array(exp_values)

            # use dataset helper to convert to final array format
            explanation_obj = Explanation(
                values=exp_values,
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
