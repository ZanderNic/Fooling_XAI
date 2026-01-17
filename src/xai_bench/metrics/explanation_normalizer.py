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
            s = np.sum(np.abs(e), axis=-1, keepdims=True)
            return np.array([
                vec / s_i if s_i > 0 else vec
                for vec, s_i in zip(e, s)
            ])

        if self.mode == "l2":
            n = np.linalg.norm(e, axis=-1, keepdims=True)
            return np.array([
                vec / n_i if n_i > 0 else vec
                for vec, n_i in zip(e, n)
            ])

        raise ValueError(f"Unknown normalization mode: {self.mode}")