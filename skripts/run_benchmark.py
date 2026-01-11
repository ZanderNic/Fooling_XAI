# std lib imports
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal

# 3-party imports
from sklearn.metrics import accuracy_score

# projekt imports
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
from run_benchmark_utils import *


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
ATTACKS = [ "DistributionShiftAttack","ColumnSwitchAttack"]
EXPLAINER = [ "Shap", "Lime" ]


def run(
    dataset: BaseDataset,
    model: Literal["CNN1D", "MLP", "RF"],
    attack: Literal["DistributionShiftAttack", "ColumnSwitchAttack"],
    explainer: Literal["Shap", "Lime"],
    metric: BaseMetric,
    seed: int,
    num_samples: int  = 1000
):
    """
    
    """
    
    # get model
    model = load_model(model, dataset, seed)                          
    model.fit(dataset.X_train.values, dataset.y_train.values)   
    acc = accuracy_score(dataset.y_test.values, model.predict(dataset.X_test.values))
    
    # get explainer 
    explainer = load_explainer(explainer, dataset, seed)
    explainer.fit(dataset.X_train.values, model, dataset.features)
    
    
    # get attack
    attack = load_attack(attack_string=attack, dataset=dataset, model=model, explainer=explainer, metric=metric, seed=seed)
    
    # how many samples to get the score
    if len(dataset.X_test) <= num_samples:
        X_test = dataset.X_test 
    else:
        X_test = dataset.X_test.sample(n=num_samples)
    
    # we need to compare the distance of the real explaination and the attacked one 
    X_adv, t_generate = timed_call(attack.generate, X_test)

    # generate explanation for real dataset X_test and X_adv
    x_real_exp = explainer.explain(X_test)
    x_adv_exp = explainer.explain(X_adv)
    
    
    scores: Dict[str, float] = {}
    for name, MetricCls in METRICS.items():
        m = MetricCls()
        scores[name] = float(m(x_real_exp, x_adv_exp))
    
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
        "timing": {
            "attack_generate": asdict(t_generate),
        },
        "scores": scores,
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS.keys())
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("attack", choices=ATTACKS)
    parser.add_argument("explainer", choices=EXPLAINER)
    parser.add_argument("metric", choices=METRICS.keys())
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Controls all stochastic components (model init, explainer sampling, attacks).")

    args = parser.parse_args()
    
    result = run(dataset=DATASETS[args.dataset](), model=args.model, attack=args.attack, explainer=args.explainer,metric=METRICS[args.metric](), seed = args.seed)

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
