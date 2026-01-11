# std-lib imports
from __future__ import annotations
import tempfile
from pathlib import Path

# 3-party imports 
import numpy as np
import pandas as pd

# Projekt imports 
from xai_bench.datasets.heart_dataset import HeartDataset
from xai_bench.explainer.lime_explainer import LimeTabularAdapter
from xai_bench.explainer.shap_explainer import ShapAdapter
from xai_bench.models.random_forest import SKRandomForest


def make_synth_heart_csv(path: Path, n: int = 250, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "age": rng.integers(29, 77, size=n),
            "sex": rng.integers(0, 2, size=n),
            "trestbps": rng.integers(90, 200, size=n),
            "chol": rng.integers(120, 500, size=n),
            "fbs": rng.integers(0, 2, size=n),
            "thalach": rng.integers(70, 210, size=n),
            "exang": rng.integers(0, 2, size=n),
            "oldpeak": rng.normal(1.0, 1.0, size=n).clip(0, 6),

            "cp": rng.integers(0, 4, size=n),
            "restecg": rng.integers(0, 3, size=n),
            "slope": rng.integers(0, 3, size=n),
            "thal": rng.integers(0, 3, size=n),
        }
    )

    logit = (
        0.03 * (df["age"] - 50)
        + 0.01 * (df["chol"] - 240)
        + 0.5 * (df["cp"] == 3).astype(float)
        - 0.4 * (df["thalach"] > 150).astype(float)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    df["condition"] = (rng.random(n) < p).astype(int)

    df.to_csv(path, index=False)
    return path


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "heart.csv"
        make_synth_heart_csv(csv_path, n=300, seed=0)


        dataset = HeartDataset(str(csv_path), test_size=0.25, random_state=0, stratify=True)

        raw_order = [c for c in dataset.df_raw.columns if c != dataset.target]

        print("Raw columns (no target):")
        print(raw_order)
        print()


        print("=== CHECK: original feature order ===")
        if hasattr(dataset, "original_feature_order") and dataset.original_feature_order is not None:
            print("dataset.original_feature_order:")
            print(dataset.original_feature_order)
            assert_true(dataset.original_feature_order == raw_order,
                        "original_feature_order != raw_order (df_raw without target).")
            feature_order = dataset.original_feature_order
        else:
            print("WARNING: dataset.original_feature_order is missing. Order is NOT guaranteed.")
            feature_order = raw_order  

        print("OK\n")


        print("=== CHECK: feature_mapping completeness ===")
        for f in feature_order:
            assert_true(f in dataset.feature_mapping, f"Missing mapping entry for original feature '{f}'")
            sub = dataset.feature_mapping[f]
            assert_true(isinstance(sub, list) and len(sub) >= 1,
                        f"feature_mapping['{f}'] must be a non-empty list.")
        print("OK\n")


        print("=== FIT: RandomForest ===")
        model = SKRandomForest(
            task="classification",
            n_estimators=60,
            random_state=0,
            max_depth=6,
        )
        model.fit(dataset.X_train.values, dataset.y_train.values)
        print("OK\n")

        print("=== LIME: fit + explain ===")
        lime = LimeTabularAdapter(dataset=dataset, num_samples=800, random_state=0)
        lime.fit(reference_data=dataset.X_train.values, model=model, features=dataset.features)

        x = dataset.X_test.values[0]
        target = 1
        lime_attr = lime.explain(x, target=target)

        assert_true(isinstance(lime_attr, np.ndarray) and lime_attr.ndim == 1,
                    "LIME attribution must be 1D numpy array.")
        assert_true(len(lime_attr) == len(feature_order),
                    f"LIME attribution length {len(lime_attr)} != number of original features {len(feature_order)}.")
        assert_true(np.any(np.abs(lime_attr) > 1e-10),
                    "LIME returned all zeros (unexpected).")

        print(f"LIME OK: attr shape = {lime_attr.shape}\n")


        print("=== SHAP: fit + explain ===")
        shap_exp = ShapAdapter(dataset=dataset, nsamples=200, background_size=50, random_state=0)
        shap_exp.fit(reference_data=dataset.X_train.values, model=model, features=dataset.features)

        shap_attr = shap_exp.explain(x, target=target)

        assert_true(isinstance(shap_attr, np.ndarray) and shap_attr.ndim == 1,
                    "SHAP attribution must be 1D numpy array.")
        assert_true(len(shap_attr) == len(feature_order),
                    f"SHAP attribution length {len(shap_attr)} != number of original features {len(feature_order)}.")
        assert_true(np.any(np.abs(shap_attr) > 1e-10),
                    "SHAP returned all zeros (unexpected).")

        print(f"SHAP OK: attr shape = {shap_attr.shape}\n")


        print("=== REPORT (top abs attributions) ===")
        def topk(names, vals, k=8):
            idx = np.argsort(np.abs(vals))[::-1][:k]
            return [(names[i], float(vals[i])) for i in idx]

        print("Top LIME:")
        for name, v in topk(feature_order, lime_attr, k=8):
            print(f"  {name:10s}  {v:+.6f}")
        print()

        print("Top SHAP:")
        for name, v in topk(feature_order, shap_attr, k=8):
            print(f"  {name:10s}  {v:+.6f}")
        print()

        print("ALL CHECKS PASSED.")
