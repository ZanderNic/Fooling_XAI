# std lib imports
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# 3-party imports
import pandas as pd
import numpy as np
from rich.progress import track

# projekt imports
from xai_bench.base import BaseAttack, BaseDataset, BaseExplainer, BaseMetric, BaseModel
from xai_bench.console import console


##### Utils 
def load_model(
    model_string: str,
    dataset: BaseDataset,
    seed: int
) -> BaseModel:
    """
        Instantiate and return a model according to the selected model string and dataset task.
    """
    assert dataset.task is not None, "Dataset task not defined"
    assert dataset.X_train is not None, "Dataset has no X-train"
    if model_string == "CNN1D":
        from xai_bench.models.cnn_1d import TorchCNN1D
        return TorchCNN1D(
            task=dataset.task,
            num_classes = dataset.num_classes,
            in_channels = 1,
            seq_len = dataset.X_train.shape[1],
            seed = seed
        )

    if model_string == "MLP":
        from xai_bench.models.mlp import TorchMLP
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
        from xai_bench.models.random_forest import SKRandomForest
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
        from xai_bench.explainer.lime_explainer import LimeTabularAdapter
        return LimeTabularAdapter(
            dataset = dataset,
            num_samples = 5000,
            random_state= seed
        )

    if explainer_string == "Shap":
        from xai_bench.explainer.shap_explainer import ShapAdapter
        return ShapAdapter(
            dataset = dataset,
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
    epsilon: float
) -> BaseAttack:
    """
        Instantiate and return an attack according to the selected attack string.
    """
    assert dataset.task is not None, "Dataset problem .()"

    if attack_string == "RandomWalkAttack":
        from xai_bench.attacks.random_walk_attack import RandomWalkAttack
        attack = RandomWalkAttack(
            dataset=dataset,
            model=model,
            explainer=explainer,
            metric=metric,
            epsilon=epsilon,
            seed=seed,
            task=dataset.task,
            num_steps=100
        )
        
        attack.fit()
        return attack
    
    if attack_string == "RandomWalkWithMemoryAttack":
        from xai_bench.attacks.random_walk_with_memory_attack import RandomWalkWithMemoryAttack
        attack = RandomWalkWithMemoryAttack(
            dataset=dataset,
            model=model,
            explainer=explainer,
            metric=metric,
            epsilon=epsilon,
            seed=seed,
            task=dataset.task,
            num_runs=10,
            num_steps=100
        )
        
        attack.fit()
        return attack
    
    if attack_string == "MonteCarloAttack":
        from xai_bench.attacks.monte_carlo_attack import MonteCarloAttack
        attack = MonteCarloAttack(
            dataset=dataset,
            model=model,
            explainer=explainer,
            metric=metric,
            epsilon=epsilon,
            seed=seed,
            task=dataset.task,
            num_candidates=100,
            max_distance=0.1
        )
        
        attack.fit()
        return attack
    
    if attack_string == "TrainLookupAttack":
        from xai_bench.attacks.train_lookup_attack import TrainLookupAttack
        attack = TrainLookupAttack(
            dataset=dataset,
            model=model,
            explainer=explainer,
            metric=metric,
            epsilon=epsilon,
            seed=seed,
            task=dataset.task,
            max_candidates=100
        )
        
        attack.fit()
        return attack

    if attack_string == "ColumnSwitchAttack":
        from xai_bench.attacks.switch_column_attack import ColumnSwitchAttack
        attack =  ColumnSwitchAttack(
            model=model,
            task= dataset.task,
            epsilon=epsilon,
            dataset=dataset
            #explainer=explainer,
            #metric=metric,
            #random_state=seed,
        )
        
        attack.fit(dataset=dataset, n_switches=5, max_tries=1000, numerical_only=True)
        return attack
    
    if attack_string == "DataPoisoningAttack":
        from xai_bench.attacks.data_poisoning_attack import DataPoisoningAttack
        attack =  DataPoisoningAttack(
            dataset=dataset,
            model=model,
            explainer=explainer,
            task= dataset.task,
            epsilon=epsilon,
            random_state=seed
        )

        attack.fit(
            N_GEN=120,
            N_POP=20,
            N_SAMPLE=10,
            INIT_MUTATION_RATE=1.0,
            INIT_STD=0.2,
            P_ELITE=0.05,
            P_COMBINE=0.1,
            DRIFT_THRESHOLD=0.3,
            DRIFT_CONFIDENCE=0.95,
            EARLY_STOPPING_PATIENCE=7,
            EXPLAINER_NUM_SAMPLES=150,
            EVOLUTION_DATA_NUM_SAMPLES=200
        )

        return attack
    
    if attack_string == "GreedyHillClimb":
        from xai_bench.attacks.greedy_hill_climb import GreedyHillClimb
        attack = GreedyHillClimb(
            dataset=dataset,
            model=model,
            explainer=explainer,
            metric=metric,
            epsilon=epsilon,
            seed=seed,
            task=dataset.task,
            num_climbs=100,
            num_derections=100,
            max_trys=1,
            step_len=0.001,
            proba_numeric=0.7,
        )
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

def get_attack_success(X:np.ndarray,X_adv:np.ndarray) -> tuple[np.ndarray, int, float]: 
    assert X.shape == X_adv.shape, "Must have same shape"
    unchanged = (X == X_adv).all(axis=1)
    success = ~unchanged

    num_success = int(success.sum())
    success_rate = float(num_success / len(X))

    return success, num_success, success_rate

def calculate_metrics(X_exp:np.ndarray, X_adv_exp:np.ndarray, METRICS:dict)->dict:
    explain_scores: Dict[str, dict] = {}
    for name, MetricCls in track(
        METRICS.items(), description="Caluclating Explaination scores", transient=True
    ):
        m: BaseMetric = MetricCls()
        s = m.compute(X_exp, X_adv_exp)
        explain_scores[name] = {"mean": float(s.mean()), "std": float(s.std())}
    return explain_scores