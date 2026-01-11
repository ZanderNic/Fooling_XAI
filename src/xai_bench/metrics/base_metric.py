import numpy as np
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Abstract base class for explanation distance metrics.

    This class defines the interface for metrics that compute a distance
    between two explanation vectors (e.g., feature importances).

    Attributes
    ----------
    name : str
        Name of the metric.

    Methods
    -------
    compute(e1: np.ndarray, e2: np.ndarray) -> float
        Compute the distance between two explanation vectors.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
        """
        Compute the distance between two explanation vectors.

        Parameters
        ----------
        e1 : np.ndarray
            First explanation vector (feature importances).
        e2 : np.ndarray
            Second explanation vector (feature importances).

        Returns
        -------
        float
            Distance or dissimilarity between e1 and e2.
        """
        pass