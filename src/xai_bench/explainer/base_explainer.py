# std lib imports
from __future__ import annotations
from typing import Any, Callable, Literal, Optional

# 3-party imports 
import numpy as np

# Options
TaskType = Literal["classification", "regression"]
OutputSpace = Literal["model", "raw"]
#

class BaseExplainer:
    """
        Abstract base class for all explanation methods used in the benchmark.

        This class defines the minimal interface that every explainer (e.g. LIME, SHAP)
        must implement in order to be compatible with the evaluation pipeline,
        attack implementations, and metrics.

        The goal of this base class is to enforce a uniform contract:
        - how explainers are initialized (`fit`)
        - how explanations for individual samples are generated (`explain`)
        - what information an explanation must contain (via the `Explanation` object)

        Concrete explainers must subclass this class and implement the `explain` method.
    
    """

    def fit(
        self, 
        reference_data: np.ndarray, 
        model, 
        features
    ) -> None:
        """  
            Initialize the explainer with all global information required to produce
            local explanations.

            This method is called once before explanations are generated. It provides
            the explainer with:
            - reference_data (e.g. for sampling or expectations),
            - a model defining the model to be explained,
            - a feature specification describing the input feature space.

            The exact use of the reference_data depends on the explainer:
            - For SHAP, it typically defines the background distribution.
            - For LIME, it defines the data distribution used for perturbations.
            - Some explainers may ignore it entirely.

            Args:
                reference_data (np.ndarray):
                    reference_data expressed in the same feature space as the model input.

                model:
                    A model specification or wrapper object that exposes a scalar
                    prediction function to be explained (e.g. probability for a
                    target class or regression output).

                features:
                    A feature space specification describing the model input features,
                    including feature names and optional mappings between model-space
                    and raw feature representations.
        """
        raise NotImplementedError


    def explain(
        self, 
        x: np.ndarray, 
        target: Optional[int] = None
    ) -> np.array:
        """
            Generate a local explanation for a single input sample.

            This method computes feature attributions explaining the model's prediction
            for the given input `x`. The explanation must be returned in a standardized
            `Explanation` object to ensure compatibility with attacks and metrics.

            For classification tasks, the explanation is typically computed with respect
            to a specific target class. If `target` is None, the explainer should default
            to the model's predicted class for `x`.

            For regression tasks, `target` is ignored.

            Args:
                x (np.ndarray):
                    One-dimensional input sample with shape (d,), expressed in the
                    model input feature space.

                target (Optional[int]):
                    Index of the class to be explained for classification models.
                    If None, the predicted class should be used by default.

            Returns:
                Explanation:
                    A standardized explanation object containing:
                    - a dense vector of feature attributions,
                    - optional base or reference value,
                    - the explained target (if applicable),
                    - metadata such as runtime or model query count.

            Raises:
                NotImplementedError:
                    If the method is not implemented by a subclass.
        """
        raise NotImplementedError

