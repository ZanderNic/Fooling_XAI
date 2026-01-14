from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np





class BaseModel(ABC):
    """
    Minimal base interface for benchmark models.

    This interface is intentionally strict to avoid ambiguous behavior in
    XAI benchmarks and adversarial evaluation.

    Supported tasks:
        - "classification"
        - "regression"

    Task-specific contracts:

    Classification models:
        - must implement:
            * predict_proba(X) -> np.ndarray of shape (n, C)
        - must NOT implement:
            * predict_scalar
        - predict(X) returns the predicted class labels via argmax over probabilities

    Regression models:
        - must implement:
            * predict_scalar(X) -> np.ndarray of shape (n,)
        - must NOT implement:
            * predict_proba
        - predict(X) returns the scalar predictions directly

    No implicit conversions:
        - No confidence scores
        - No automatic target selection
        - No task switching at runtime
    """

    def __init__(self, task: Literal["classification","regression"]):
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression' but is {task}")
        self.task:Literal["classification","regression"] = task


    @abstractmethod
    def fit(self, X, y) -> "BaseModel":
        """
        Fit the model on training data.

        Args:
            X:
                Input samples in the model's expected input format.
            y:
                Target values (class labels for classification, scalars for regression).

        Returns:
            self
        """
        ...


    def predict(self, X) -> np.ndarray:
        """
        Unified prediction interface.

        Behavior depends strictly on the task:

        Classification:
            - Uses predict_proba(X)
            - Returns argmax over class probabilities
            - Output shape: (n,)
            - dtype: integer class indices

        Regression:
            - Uses predict_scalar(X)
            - Returns scalar predictions directly
            - Output shape: (n,)
            - dtype: float

        Raises:
            NotImplementedError:
                If the required task-specific method is not implemented.
        """
        if self.task == "classification":
            proba = self.predict_proba(X)
            if proba.ndim != 2:
                raise ValueError(
                    "predict_proba must return an array of shape (n, C)"
                )
            return np.argmax(proba, axis=1)

        # regression
        return self.predict_scalar(X)

    def predict_raw(self, X) -> np.ndarray:
        """
        Direct prediction interface without task-based post-processing.

        Behavior depends strictly on the task:

        Classification:
            - Calls predict_proba(X)
            - Returns class probabilities directly
            - Output shape: (n, C)

        Regression:
            - Calls predict_scalar(X)
            - Returns scalar predictions directly
            - Output shape: (n,)
        """
        if self.task == "classification":
            return self.predict_proba(X)
        # regression
        return self.predict_scalar(X)

    def predict_proba(self, X) -> np.ndarray:
        """
        Return class probabilities for classification models.

        Contract:
            - Only valid if task == "classification"
            - Must return an array of shape (n, C)
            - Rows must sum to 1 (probability simplex)

        Raises:
            NotImplementedError:
                If called on a regression model or not overridden by subclass.
        """
        raise NotImplementedError(
            "predict_proba is only defined for classification models."
        )

    def predict_scalar(self, X) -> np.ndarray:
        """
        Return scalar predictions for regression models.

        Contract:
            - Only valid if task == "regression"
            - Must return a 1D array of shape (n,)
            - No implicit reshaping or aggregation

        Raises:
            NotImplementedError:
                If called on a classification model or not overridden by subclass.
        """
        raise NotImplementedError(
            "predict_scalar is only defined for regression models."
        )
