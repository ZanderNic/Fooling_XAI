import numpy as np

class ExplanationNormalizer:
    def __init__(self, mode: str = "l1"):
        """
        mode:
          - "l1": sum(|e|) = 1
          - "l2": ||e|| = 1
          - "none":
        """
        self.mode = mode

    def __call__(self, e: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return e

        if self.mode == "l1":
            s = np.sum(np.abs(e))
            return e / s if s > 0 else e

        if self.mode == "l2":
            n = np.linalg.norm(e)
            return e / n if n > 0 else e

        raise ValueError(f"Unknown normalization mode: {self.mode}")