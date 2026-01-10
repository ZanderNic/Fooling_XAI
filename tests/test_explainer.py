from __future__ import annotations

import numpy as np

# import your adapters (adjust paths!)
from xai_bench.explainer.lime_explainer import LimeTabularAdapter
from xai_bench.explainer.shap_explainer import ShapAdapter
from xai_bench.explainer.base_explainer import Features


import sys, os
sys.path.append(os.path.abspath("src"))



class NumpyLinearModel:
    def __init__(self, W: np.ndarray, b: np.ndarray, task: str):
        self.W = np.asarray(W, dtype=float)  # (C,d) for clf, (d,) for reg
        self.b = np.asarray(b, dtype=float)  # (C,) for clf, () or (1,) for reg
        self.task = task

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = X @ self.W.T + self.b  # (n,C)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_scalar(self, X: np.ndarray, target=None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        y = X @ self.W + float(self.b)
        return np.asarray(y, dtype=float).reshape(-1)


def test_classification():
    rng = np.random.default_rng(0)
    n, d, C = 400, 6, 3
    Xref = rng.normal(size=(n, d))

    W = rng.normal(size=(C, d))
    b = rng.normal(size=(C,))
    model = NumpyLinearModel(W=W, b=b, task="classification")
    features = Features([f"f{i}" for i in range(d)])

    x = Xref[0]
    target = 1

    lime = LimeTabularAdapter(num_samples=2000, num_features=None, random_state=0)
    shap_expl = ShapAdapter(nsamples=500, background_size=200, random_state=0)

    lime.fit(Xref, model, features)
    shap_expl.fit(Xref, model, features)

    e_lime = lime.explain(x, target=target)
    e_shap = shap_expl.explain(x, target=target)

    print("=== CLASSIFICATION ===")
    print("x shape:", x.shape)
    print("LIME attributions shape:", e_lime.attributions.shape)
    print("SHAP attributions shape:", e_shap.attributions.shape)
    print("SHAP base_value:", e_shap.base_value)
    print("LIME target:", e_lime.target, " | SHAP target:", e_shap.target)
    print("LIME first 3:", e_lime.attributions)
    print("SHAP first 3:", e_shap.attributions)
    print()


def test_regression():
    rng = np.random.default_rng(1)
    n, d = 400, 6
    Xref = rng.normal(size=(n, d))

    W = rng.normal(size=(d,))
    b = np.array(0.2)
    model = NumpyLinearModel(W=W, b=b, task="regression")
    features = Features([f"f{i}" for i in range(d)])

    x = Xref[0]

    lime = LimeTabularAdapter(num_samples=2000, num_features=None, random_state=0)
    shap_expl = ShapAdapter(nsamples=500, background_size=200, random_state=0)

    lime.fit(Xref, model, features)
    shap_expl.fit(Xref, model, features)

    e_lime = lime.explain(x, target=None)
    e_shap = shap_expl.explain(x, target=None)

    print("=== REGRESSION ===")
    print("x shape:", x.shape)
    print("LIME attributions shape:", e_lime.attributions.shape)
    print("SHAP attributions shape:", e_shap.attributions.shape)
    print("SHAP base_value:", e_shap.base_value)
    print("LIME target:", e_lime.target, " | SHAP target:", e_shap.target)
    print("LIME first 3:", e_lime.attributions)
    print("SHAP first 3:", e_shap.attributions)
    print()


if __name__ == "__main__":
    test_classification()
    test_regression()