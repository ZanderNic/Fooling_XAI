# std
import sys
from pathlib import Path

# fix path to avoid any module errors
sys.path.insert(0, (Path(__file__).parent.parent / "src").__str__())
import argparse

from xai_bench.base import BaseDataset, BaseModel, BaseMetric, BaseExplainer, BaseAttack

# datasets
from xai_bench.datasets.heart_dataset import HeartDataset
from xai_bench.datasets.heart_failure import Heart_Failure
from xai_bench.datasets.credit_dataset import CreditDataset
from xai_bench.datasets.prisoners import PrisoneresDataset

# models
from xai_bench.models.cnn_1d import TorchCNN1D
from xai_bench.models.mlp import TorchMLP
from xai_bench.models.random_forest import RandomForestClassifier, RandomForestRegressor

# attacks
from xai_bench.attacks.distribution_shift_attack import DistributionShiftAttack
from xai_bench.attacks.switch_column_attack import ColumnSwitchAttack

# explainer
from xai_bench.explainer.lime_explainer import LimeTabularAdapter
from xai_bench.explainer.shap_explainer import ShapAdapter

# metrics
from xai_bench.metrics.cosine_metric import CosineMetric
from xai_bench.metrics.spearmen_metric import SpearmanMetric
from xai_bench.metrics.wasserstein_metric import WassersteinMetric

# evaluator
from xai_bench.attack_evaluator import AttackEvaluator
from typing import Literal

DATASETS = {
    "heart-uci": HeartDataset,
    "heart-failure": Heart_Failure,
    "credit": CreditDataset,
    "prisoners": PrisoneresDataset,
}
MODELS = ["CNN1D", "MLP", "RFClass", "RFReg"]
ATTACKS = ["DistributionShiftAttack", "ColumnSwitchAttack"]
EXPLAINER = ["Shap", "Lime"]
METRICS = {
    "Cosine": CosineMetric,
    "Spearman": SpearmanMetric,
    "Wasserstein": WassersteinMetric,
}


def run(
    dataset: BaseDataset,
    model: Literal["CNN1D", "MLP", "RFClass", "RFReg"],
    attack: Literal["DistributionShiftAttack", "ColumnSwitchAttack"],
    explainer: Literal["Shap", "Lime"],
    metric: BaseMetric
):
    # run config
    # evaluate/summarize reults and save them somewhere
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS.keys())
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("attack", choices=ATTACKS)
    parser.add_argument("explainer", choices=EXPLAINER)
    parser.add_argument("metric", choices=METRICS.keys())

    args = parser.parse_args()

    print(args)

    run(dataset=DATASETS[args.dataset], model=args.model, attack=args.attack, explainer=args.explainer,metric=METRICS[args.metric])
