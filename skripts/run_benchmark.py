from __future__ import annotations
import sys
from pathlib import Path
from rich import print as rich_print
from rich.progress import track
from rich.panel import Panel
from rich.align import Align
import time 
from itertools import product

try:
    import xai_bench  # noqa: F401
except ModuleNotFoundError:
    # in case module not correcltyloaded hardcode path
    rich_print(
        f"[#aa0000][bold]xai_bench not found![/bold] Adding [italic #222222]{(Path(__file__).parent.parent / 'src').__str__()}[/italic #222222] into python path.[/#aa0000]"
    )
    sys.path.insert(0, (Path(__file__).parent.parent / "src").__str__())

# std lib imports
import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

# 3-party imports
import numpy as np


# projekt imports
from xai_bench.console import console, RUN_TEXT, TC
from xai_bench.base import BaseDataset, BaseMetric
from xai_bench.stat_collector import StatCollector

# datasets
from xai_bench.datasets.credit_dataset import CreditDataset
from xai_bench.datasets.heart_dataset import HeartDataset
from xai_bench.datasets.prisoners import PrisoneresDataset
from xai_bench.datasets.housing import HousingDataset
from xai_bench.datasets.covtype_forest import ForestDataset

# metrics
from xai_bench.metrics.cosine_metric import CosineMetric
from xai_bench.metrics.spearmen_metric import SpearmanMetric
from xai_bench.metrics.l2_metric import L2Metric
from xai_bench.metrics.kendall_tau_metric import KendallTauMetric
from xai_bench.metrics.distortion_metric import DistortionMetric

# utils imports
from run_benchmark_utils import (
    load_model,
    load_explainer,
    load_attack,
    now_utc_iso,
    timed_call,
    get_attack_success,
    calculate_metrics
)


DATASETS = {
    "heart_uci": HeartDataset,
    "housing": HousingDataset,
    "credit": CreditDataset,
    "prisoners": PrisoneresDataset,
    "forest": ForestDataset
}

METRICS = {
    "L2": L2Metric,
    "Cosine": CosineMetric,
    "Spearman": SpearmanMetric,
    # "KendallTau": KendallTauMetric,
    # "Distortion": DistortionMetric
}

MODELS = ["CNN1D", "MLP", "RF"]
ATTACKS = ["RandomWalkAttack", "RandomWalkWithMemoryAttack", "MonteCarloAttack", "TrainLookupAttack", "ColumnSwitchAttack", "DataPoisoningAttack", "GreedyHillClimb"]
EXPLAINER = ["Shap", "Lime"]


def run(
    dataset: BaseDataset,
    model_name: Literal["CNN1D", "MLP", "RF"],
    attack_name: Literal["RandomWalkAttack", "RandomWalkWithMemoryAttack", "MonteCarloAttack", "TrainLookupAttack", "ColumnSwitchAttack", "DataPoisoningAttack"],
    explainer_name: Literal["Shap", "Lime"],
    metric: BaseMetric,
    seed: int,
    num_samples: int = 1000,
    epsilon: float = 0.05,
    train_samples: Optional[int]=None
):
    """ """

    if train_samples is not None:
        dataset.X_test = dataset.X_test[:train_samples] # type: ignore
        dataset.X_test_scaled = dataset.X_test_scaled[:train_samples] # type: ignore
        dataset.X_train = dataset.X_train[:train_samples] # type: ignore
        dataset.X_train_scaled = dataset.X_train_scaled[:train_samples] # type: ignore
        dataset.y_test = dataset.y_test[:train_samples] # type: ignore
        dataset.y_train = dataset.y_train[:train_samples] # type: ignore

    # load model
    with console.status(f"{TC} Loading model: {model_name}", spinner="shark"):
        model = load_model(model_name, dataset, seed)
    console.print(f"{RUN_TEXT} Loaded model: ", model_name)

    # fit model
    assert dataset.X_train_scaled is not None and dataset.y_train is not None, (
        "Something went wrong with the dataset"
    )
    with console.status(f"{TC} Fitting Model", spinner="shark"):
        model.fit(dataset.X_train_scaled.values, dataset.y_train.values)
    console.print(f"{RUN_TEXT} Fitted Model ")

    # predict on test and calucalte accuracy
    assert dataset.y_test is not None and dataset.X_test_scaled is not None, (
        "Something went wrong with the dataset"
    )
    with console.status(f"{TC} Calculating accuracy", spinner="shark"):
        acc = model.score(
            dataset.X_test_scaled.values, dataset.y_test.values
        )
    console.print(f"{RUN_TEXT} Calculated accuracy")

    # load explainer
    with console.status(f"{TC} Loading explainer", spinner="shark"):
        explainer = load_explainer(explainer_name, dataset, seed)
    console.print(f"{RUN_TEXT} Loaded Explainer")

    # fit explainer
    assert dataset.features is not None, "Something went wrong with the dataset"
    with console.status(f"{TC} Fitting explainer", spinner="shark"):
        _, t_exp_fit = timed_call(
            explainer.fit, dataset.X_train_scaled.values, model, dataset.features
        )
    console.print(f"{RUN_TEXT} Fitted Explainer")

    # get attack
    with console.status(f"{TC} Loading attack", spinner="shark"):
        attack, t_attack_fit = timed_call(
            load_attack,
            attack_string=attack_name,
            dataset=dataset,
            model=model,
            explainer=explainer,
            metric=metric,
            seed=seed,
            epsilon=epsilon
        )
    console.print(f"{RUN_TEXT} Loaded Attack")

    # how many samples to get the score
    if len(dataset.X_test_scaled) <= num_samples:
        X_test = dataset.X_test_scaled.values
        y_test = dataset.y_test.values
    else:
        sample_indices = np.random.RandomState(seed).choice(
            dataset.X_test_scaled.shape[0], size=num_samples, replace=False
        )
        X_test = dataset.X_test_scaled.iloc[sample_indices].values
        y_test = dataset.y_test.iloc[sample_indices].values


    with console.status(f"{TC} Generate attack", spinner="shark"):
        X_adv, t_generate = timed_call(attack.generate, X_test)
    console.print(f"{RUN_TEXT} Generated Attack")

    with console.status(f"{TC} Calculating attack accuracy  ", spinner="shark"):
        predict_accuracy  = model.score(
            X_adv, y_test
        )
    console.print(f"{RUN_TEXT} Calculated attack accuracy")

    with console.status(f"{TC} Calculating model fidelity after attack  ", spinner="shark"):
        model_attack_fidelity  = model.score(
            X_adv, model.predict(X_test)
        )
    console.print(f"{RUN_TEXT} Calculated attack accuracy")

    # generate explanation for real dataset X_test and X_adv
    with console.status(f"{TC} Explaining real X", spinner="shark"):
        x_real_exp = explainer.explain(X_test)

    with console.status(f"{TC} Explaining adversarial X", spinner="shark"):
        x_adv_exp = explainer.explain(X_adv)
    console.print(f"{RUN_TEXT} All X explained")

    # get attack success
    mask, succ_count, succ_rate = get_attack_success(X_test, X_adv)

    with console.status(f"{TC} Calcualting Scores on ALL data and only successfull attacks", spinner="shark"):
        explain_scores_all = calculate_metrics(x_real_exp,x_adv_exp,METRICS)
        if succ_count>1:
            explain_scores_on_success_only = calculate_metrics(x_real_exp[mask],x_adv_exp[mask],METRICS)
        else:
            explain_scores_on_success_only = {"No metrics possible": "No successfull attacks"}
    console.print(f"{RUN_TEXT} All explaination scores calculated")

    stats = StatCollector.collect(model,attack,explainer)
    console.print(stats[0])

    result = {
        "meta": {
            "timestamp_utc": now_utc_iso(),
            "seed": seed,
            "dataset": dataset.__class__.__name__,
            "model": model.__class__.__name__,
            "model_task": model.task,
            "attack": attack.__class__.__name__,
            "explainer": explainer.__class__.__name__,
            "selected_metric_for_attack": metric.__class__.__name__,
            "num_samples": X_test.shape[0],
        },
        "timing": {
            "explainer_fit": asdict(t_exp_fit),
            "attack_fit": asdict(t_attack_fit),
            "attack_generate": asdict(t_generate)
        },
        "results":{
            "accuracy": acc, # accuracy of model prediction on X test
            "attack_accuracy": predict_accuracy,  # acucracy of model preduction on attacked X test
            "model_attack_fidelity": model_attack_fidelity,  # fidelity between model prediction on X test and attacked X test
            "attack_success_rate": succ_rate,
            "attack_success_count": succ_count,
        },
        "explain_scores_on_all": explain_scores_all,
        "explain_scores_on_success_only": explain_scores_on_success_only,
        "stats":stats[1]
    }

    return result


if __name__ == "__main__":
    console.print("Checking Arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS.keys())
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("attack", choices=ATTACKS)
    parser.add_argument("explainer", choices=EXPLAINER)
    # parser.add_argument("metric", choices=METRICS.keys())
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Controls all stochastic components (model init, explainer sampling, attacks).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Num samples from the test set that are used for the evaluation",
    )
    parser.add_argument(
        "-s", "--smoke-test",
        action='store_true',
        help= "Run a smoke test over all available datatsets/attacks/explainers/metrics and save overview result. Will ignore all other parameters."
    )

    args = parser.parse_args()
    if args.smoke_test:
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
        console.print(Align.center(f"Over the parameters:  {list(DATASETS.keys())},{['L2']},{MODELS}, {ATTACKS}, {EXPLAINER}"),style="bold red")
        console.print(Align.center(f"Using only [bold cyan]{num_samples}[/bold cyan] samples."),style="bold red")
        result_dir = Path(f"./results/smoke_test_{time.time()}")
        result_dir.mkdir(parents=True, exist_ok=True)
        for dataset, metric, model, attack, explainer in track(product(DATASETS.keys(),["L2"],MODELS,ATTACKS,EXPLAINER),description="Going through all settings",total=len(DATASETS)*len(METRICS)*len(MODELS)*len(ATTACKS)*len(EXPLAINER), console=console):
            # print settings
            p = Panel(f"Current Paramters: {dataset} - {metric} - {model} - {attack} - {explainer}",style="cyan",expand=False)
            console.print(Align.center(p))
            # run run
            result = run(
                dataset=DATASETS[dataset](),
                model_name=model, # type: ignore
                attack_name=attack, # type: ignore 
                explainer_name=explainer, # type: ignore
                metric=METRICS["L2"](),
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
        exit(0)
    else:
        console.print(f"[#69db88][RUN][/#69db88]{TC} Starting new run with: [/]", args)
        result = run(
            dataset=DATASETS[args.dataset](),
            model_name=args.model,
            attack_name=args.attack,
            explainer_name=args.explainer,
            metric=METRICS["L2"](),
            seed=args.seed,
            num_samples=args.num_samples,
        )

        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        filename = (
            f"{args.dataset}__"
            f"{args.model}__"
            f"{args.explainer}__"
            f"{args.attack}__"
            f"seed{args.seed}__"
            f"{timestamp}.json"
        )

        out_path = results_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        console.print(f"[bold cyan][OK][/] Results saved to: [italic #9c9c9c]{out_path}[/]",highlight=False)
