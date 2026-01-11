# std
import argparse

# base
from xai_bench.base import BaseDataset, BaseModel, BaseMetric, BaseExplainer, BaseAttack
# datasets
from xai_bench.datasets import HeartDataset, Heart_Failure, CreditDataset, PrisoneresDataset
# models

# attacks
from xai_bench.attacks import DistributionShiftAttack, ColumnSwitchAttack
# explainer
from xai_bench.explainer import ShapAdapter, LimeTabularAdapter
# metrics
from xai_bench.metrics import CosineMetric, SpearmanMetric, WassersteinMetric


def run(dataset:BaseDataset):
    pass


if __name__=="__main__":
    DATASETS = {
        "heart-uci" : HeartDataset,
        "heart-failure": Heart_Failure,
        "credit": CreditDataset,
        "prisoners": PrisoneresDataset
    }
    MODELS = {

    }

    pass