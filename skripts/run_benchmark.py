from __future__ import annotations
import sys
from pathlib import Path
from rich import print as rich_print
from rich.progress import track

try:
    import xai_bench
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
from typing import Dict, Literal

# 3-party imports
from sklearn.metrics import accuracy_score
import numpy as np


# projekt imports
from xai_bench.console import console, RUN_TEXT, TC
from xai_bench.base import BaseDataset, BaseMetric

# datasets
from xai_bench.datasets.credit_dataset import CreditDataset
from xai_bench.datasets.heart_dataset import HeartDataset
from xai_bench.datasets.heart_failure import Heart_Failure
from xai_bench.datasets.prisoners import PrisoneresDataset

# metrics
from xai_bench.metrics.cosine_metric import CosineMetric
from xai_bench.metrics.spearmen_metric import SpearmanMetric
from xai_bench.metrics.wasserstein_metric import WassersteinMetric

# utils imports
from run_benchmark_utils import (
    load_model,
    load_explainer,
    load_attack,
    now_utc_iso,
    timed_call,
)


DATASETS = {
    "heart-uci": HeartDataset,
    "heart-failure": Heart_Failure,
    "credit": CreditDataset,
    "prisoners": PrisoneresDataset,
}

METRICS = {
    "Cosine": CosineMetric,
    "Spearman": SpearmanMetric,
    "Wasserstein": WassersteinMetric,
}

MODELS = ["CNN1D", "MLP", "RF"]
ATTACKS = ["DistributionShiftAttack", "ColumnSwitchAttack"]
EXPLAINER = ["Shap", "Lime"]


def run(
    dataset: BaseDataset,
    model_name: Literal["CNN1D", "MLP", "RF"],
    attack_name: Literal["DistributionShiftAttack", "ColumnSwitchAttack"],
    explainer_name: Literal["Shap", "Lime"],
    metric: BaseMetric,
    seed: int,
    num_samples: int = 1000,
):
    """ """

    # load model
    with console.status(f"{TC} Loading model: {model_name}", spinner="shark"):
        model = load_model(model_name, dataset, seed)
    console.print(f"{RUN_TEXT} Loaded model: ", model_name)

    # fit model
    assert dataset.X_train is not None and dataset.y_train is not None, (
        "Sth went wwrong at with the dataset"
    )
    with console.status(f"{TC} Fitting Model", spinner="shark"):
        model.fit(dataset.X_train.values, dataset.y_train.values)
    console.print(f"{RUN_TEXT} Fitted Model ")

    # predict on test and calucalte accuracy
    assert dataset.y_test is not None and dataset.X_test is not None, (
        "Sth went wrong wirh datatset"
    )
    with console.status(f"{TC} Calculating accuracy", spinner="shark"):
        acc = accuracy_score(
            dataset.y_test.values, model.predict(dataset.X_test.values)
        )
    console.print(f"{RUN_TEXT} Calculated accuracy")

    # load explainer
    with console.status(f"{TC} Loading explainer", spinner="shark"):
        explainer = load_explainer(explainer_name, dataset, seed)
    console.print(f"{RUN_TEXT} Loaded Explainer")

    # fit explainer
    assert dataset.features is not None, "Sth went wrong with dataset"
    with console.status(f"{TC} Fitting explainer", spinner="shark"):
        _, t_fit = timed_call(
            explainer.fit, dataset.X_train.values, model, dataset.features
        )
    console.print(f"{RUN_TEXT} Fitted Explainer")

    # get attack
    with console.status(f"{TC} Loading attack", spinner="shark"):
        attack = load_attack(
            attack_string=attack_name,
            dataset=dataset,
            model=model,
            explainer=explainer,
            metric=metric,
            seed=seed,
        )
    console.print(f"{RUN_TEXT} Loaded Attack")

    # how many samples to get the score
    if len(dataset.X_test) <= num_samples:
        X_test = dataset.X_test
    else:
        X_test = dataset.X_test.sample(n=num_samples)

    # we need to compare the distance of the real explaination and the attacked one

    with console.status(f"{TC} Generate attack", spinner="shark"):
        X_adv, t_generate = timed_call(attack.generate, X_test)
    console.print(f"{RUN_TEXT} Generated Attack")

    # generate explanation for real dataset X_test and X_adv
    with console.status(f"{TC} Explaining real X", spinner="shark"):
        x_real_exp = explainer.explain(np.asarray(X_test))
    with console.status(f"{TC} Explaining adverserial X", spinner="shark"):
        x_adv_exp = explainer.explain(X_adv)
    console.print(f"{RUN_TEXT} All X explained")

    scores: Dict[str, dict] = {}
    for name, MetricCls in track(
        METRICS.items(), description="Caluclating metrics", transient=True
    ):
        m: BaseMetric = MetricCls()
        s = m.compute(x_real_exp, x_adv_exp)
        scores[name] = {"mean": float(s.mean()), "std": float(s.std())}
    console.print(f"{RUN_TEXT} All metrics calculated")

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
            "num_samples": int(len(X_test)),
            "accuracy": acc,
        },
        "timing": {"attack_generate": asdict(t_generate), "attack_fit": asdict(t_fit)},
        "scores": scores,
    }

    return result


if __name__ == "__main__":
    console.print("Checking Arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS.keys())
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("attack", choices=ATTACKS)
    parser.add_argument("explainer", choices=EXPLAINER)
    parser.add_argument("metric", choices=METRICS.keys())
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Controls all stochastic components (model init, explainer sampling, attacks).",
    )

    args = parser.parse_args()

    console.print("[#69db88]\[RUN][/#69db88] Starting new run with: ", args)
    result = run(
        dataset=DATASETS[args.dataset](),
        model_name=args.model,
        attack_name=args.attack,
        explainer_name=args.explainer,
        metric=METRICS[args.metric](),
        seed=args.seed,
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

    print(f"[OK] Results saved to: {out_path}")
