# std lib imports
from __future__ import annotations
import json
import time
from dataclasses import  dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# 3-party imports

# projekt imports
from xai_bench.base import BaseAttack, BaseDataset, BaseExplainer, BaseMetric, BaseModel

# models
from xai_bench.models.cnn_1d import TorchCNN1D
from xai_bench.models.mlp import TorchMLP
from xai_bench.models.random_forest import SKRandomForest

# attacks
from xai_bench.attacks.distribution_shift_attack import DistributionShiftAttack
from xai_bench.attacks.switch_column_attack import ColumnSwitchAttack

# explainer
from xai_bench.explainer.lime_explainer import LimeTabularAdapter
from xai_bench.explainer.shap_explainer import ShapAdapter




##### Utils 
def load_model(
    model_string: str,
    dataset: BaseDataset,
    seed: int
) -> BaseModel:
    """
        Instantiate and return a model according to the selected model string and dataset task.
    """

    if model_string == "CNN1D":
        return TorchCNN1D(
            num_classes = dataset.num_classes,
            in_channels = 1,
            seq_len = dataset.X_train.shape[1],
            seed = seed
        )

    if model_string == "MLP":
        return TorchMLP(
            task = dataset.task,
            input_dim = dataset.X_train.shape[1],
            num_classes = dataset.num_classes,
            lr = 0.001,
            epochs = 30,
            batch_size= 256,
            seed = seed
        )

    if model_string == "RF":
        return SKRandomForest(
            task = dataset.task,
            n_estimators = 300,
            random_state = seed,
            max_depth = 30
        )

    raise ValueError(f"Unknown model type: {model_string}")


def load_explainer(
    explainer_string: str,
    dataset: BaseDataset,
    seed: int
) -> BaseExplainer:
    """
        Instantiate and return a model according to the selected model string and dataset task.
    """

    if explainer_string == "Lime":
        return LimeTabularAdapter(
            dataset = dataset,
            num_samples = 5000,
            random_state= seed
        )

    if explainer_string == "Shap":
        return ShapAdapter(
            dataset = dataset,
            num_samples = 5000,
            background_size= 5000,
            random_state= seed
        )

    raise ValueError(f"Unknown model type: {explainer_string}")


def load_attack(
    attack_string: str,
    dataset: BaseDataset,
    model: BaseModel,
    explainer: BaseExplainer,
    metric: BaseMetric,
    seed: int,
) -> BaseAttack:
    """
        Instantiate and return an attack according to the selected attack string.
    """

    if attack_string == "DistributionShiftAttack":
        attack =  DistributionShiftAttack(
            dataset=dataset,
            model=model,
            # explainer=explainer,  # TODO !!!!!!!11
            # metric=metric,
            # random_state=seed,
        )
        
        attack.fit()
        return attack


    if attack_string == "ColumnSwitchAttack":
        attack =  ColumnSwitchAttack(
            model=model,
            explainer=explainer,
            metric=metric,
            random_state=seed,
        )
        
        attack.fit(dataset=dataset, n_switches=30, max_tries=42)
        return attack

    raise ValueError(f"Unknown attack type: {attack_string}")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class Timing:
    seconds: float
    start_utc: str
    end_utc: str

def timed_call(fn, *args, **kwargs):
    start = time.perf_counter()
    start_iso = now_utc_iso()
    out = fn(*args, **kwargs)
    end = time.perf_counter()
    end_iso = now_utc_iso()
    return out, Timing(seconds=end - start, start_utc=start_iso, end_utc=end_iso)

def save_result_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


