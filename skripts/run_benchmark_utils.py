# std lib imports
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Type, Callable
import traceback
import inspect

# 3-party imports
import numpy as np
from rich.progress import track
from rich.panel import Panel
from rich.align import Align
from itertools import product

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
            num_samples = 1500,
            random_state= seed
        )

    if explainer_string == "Shap":
        from xai_bench.explainer.shap_explainer import ShapAdapter
        return ShapAdapter(
            dataset = dataset,
            background_size= 1500,
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
        if smoke_test:
            attack = RandomWalkAttack(
                dataset=dataset,
                model=model,
                explainer=explainer,
                metric=metric,
                epsilon=epsilon,
                seed=seed,
                task=dataset.task,
                num_steps=3
            )
        else:
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
        if False:
            assert dataset.numerical_features is not None, "Has to have num features"
            attack =  ColumnSwitchAttack(
                model=model,
                task= dataset.task,
                dataset=dataset,
                metric=metric,
                explainer=explainer,
                epsilon=epsilon,
                n_switches=int(len(dataset.numerical_features)*0.5),
                max_tries=10,
                numerical_only=True
            )
        else:
            assert dataset.features is not None and dataset.features.feature_names_model is not None, "Has to have features"
            attack =  ColumnSwitchAttack(
                model=model,
                task= dataset.task,
                dataset=dataset,
                metric=metric,
                explainer=explainer,
                epsilon=epsilon,
                n_switches=int(len(dataset.features.feature_names_model)*0.5),
                max_tries=50,
                numerical_only=False
            )
        
        attack.fit()
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
        if smoke_test:
            attack = GreedyHillClimb(
                dataset=dataset,
                model=model,
                explainer=explainer,
                metric=metric,
                epsilon=epsilon,
                seed=seed,
                task=dataset.task,
                num_climbs=4,
                num_derections=4,
                max_trys=1,
                step_len=0.001,
                proba_numeric=0.7,
                num_samples_explainer=10
            )
            return attack
        else:
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
        METRICS.items(), description="Caluclating Explaination scores", transient=True,console=console
    ):
        m: BaseMetric = MetricCls()
        s = m.compute(X_exp, X_adv_exp)
        explain_scores[name] = {"mean": float(s.mean()), "std": float(s.std())}
    return explain_scores


def infer_smoke_test(datasets:dict[str,Type[BaseDataset]],models:list[str],explainers:list[str],attacks:list[str])->tuple[dict[str,Type[BaseDataset]],list[str],list[str],list[str]]:
    console.rule("[bold green]Setting smoketest parameters[/]")
    console.print("")
    # ask for dataset
    ds = {k:datasets[k] for k in _ask([*datasets.keys()])}
    mo = _ask(models)
    ex = _ask(explainers)
    at = _ask(attacks)

    return ds, mo, ex, at

def _ask(things:list[str]):
    response = console.input(f"[green] Out of the following, which should be included? (seperate with ',' or press enter for all):[/]\n[#d7f5d7]{' - '.join(things)}[/]\n")
    if response=="":
        return things
    else:
        return response.split(",")


"""
Runs a smoke test on all combinations
"""
def smoke_test(run_func:Callable, datasets:dict[str,Type[BaseDataset]],metrics:dict[str,Type[BaseMetric]],models:list[str],explainers:list[str],attacks:list[str]):
    # print run smoek test
    console.print(
        Panel(
            Align.center("SMOKE TEST",vertical="middle"),
            style="bold red",
            border_style="red",
            padding=(3,10)
            )
        )
    num_samples = 2
    console.print(Align.center(f"Over the parameters:  {list(datasets.keys())},{['L2']},{models}, {attacks}, {explainers}"),style="bold red")
    console.print(Align.center(f"Using only [bold cyan]{num_samples}[/bold cyan] samples."),style="bold red")
    result_dir = Path(Path(__file__).parent.parent/f"./results/smoke_test_{time.time()}")
    result_dir.mkdir(parents=True, exist_ok=True)
    for dataset, metric, model, attack, explainer in track(product(datasets.keys(),["L2"],models,attacks,explainers),description="Going through all settings",total=len(datasets)*len(models)*len(attacks)*len(explainers), console=console):
        try:
            # print settings
            p = Panel(f"Current Paramters: {dataset} - {metric} - {model} - {attack} - {explainer}",style="cyan",expand=False)
            console.print(Align.center(p))
            # run run
            result = run_func(
                dataset=datasets[dataset](),
                model_name=model, # type: ignore
                attack_name=attack, # type: ignore 
                explainer_name=explainer, # type: ignore
                metric=metrics["L2"](),
                seed=42,
                num_samples=num_samples,
                train_samples=num_samples
            )
            # save results
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
            filename = (
                f"{dataset}__"
                f"{model}__"
                f"{explainer}__"
                f"{attack}__"
                f"seed{42}__"
                f"{timestamp}.json"
            )

            out_path = result_dir / filename
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception:
            error_file = result_dir / (f"error_{time.time_ns()}.txt")
            console.print(Panel(f"[bold red]An error occured![/]\n\nFor further inforamtion look see [lightgray italic]{error_file}[/]",style="bold red"))
            with error_file.open("w") as f:
                f.write(traceback.format_exc()+"\n\n"+f"Current Paramters: {dataset} - {metric} - {model} - {attack} - {explainer}")


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, OverflowError):
        return str(value)
    
def get_args(*objects):
    result = {}

    for obj in objects:
        cls = obj.__class__
        name = cls.__name__

        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())[1:]  # skip self

        init_args = {}
        for p in params:
            if hasattr(obj, p.name):
                value = getattr(obj, p.name)
            elif p.default is not inspect._empty:
                value = p.default
            else:
                value = None

            init_args[p.name] = _json_safe(value)

        result[name] = init_args

    return result
