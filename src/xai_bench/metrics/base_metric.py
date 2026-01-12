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

    # should be a distance, so maximising
    @abstractmethod
    def _compute(self, e1: np.ndarray, e2: np.ndarray) -> float:
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
    
    # returns all distances of all explanation vectors. Input should have shape (n, n_features). Output is (n,)
    def compute(self, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
        """
        Compute distances between corresponding explanation vectors.

        Parameters
        ----------
        e1 : np.ndarray
            Explanation vectors of shape (n, n_features).
        e2 : np.ndarray
            Explanation vectors of shape (n, n_features).

        Returns
        -------
        np.ndarray
            Array of distances of shape (n,).
        """
        return np.array([self._compute(e1[i], e2[i]) for i in range(len(e1))])
        