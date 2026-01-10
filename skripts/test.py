import os

import sys
sys.modules.keys()

from xai_bench.datasets.heart_dataset import HeartDataset
from xai_bench.models.random_forest import SKRandomForest
from xai_bench.explainer.lime_explainer import LimeTabularAdapter
from xai_bench.attacks.distribution_shift_attack import DistributionShiftAttack
from xai_bench.metrics.cosine_metric import CosineMetric
from xai_bench.attack_evaluator import AttackEvaluator

path = os.path.join("datasets", "heart.csv")
dataset = HeartDataset(path)

model = SKRandomForest("classification")

lime_explainer = LimeTabularAdapter(
    dataset=dataset
    )
lime_explainer.fit(
    reference_data=dataset.X_train.values,
    model=model,
    features=dataset.features
)

attack = DistributionShiftAttack(
    dataset=dataset,
    model=model,
    epsilon=0.1,
    max_tries=100,
    prob_tolerance=0.01
)

metric = CosineMetric()

evaluator = AttackEvaluator(
    dataset=dataset,
    model=model,
    explainer=lime_explainer,
    attack=attack,
    metric=metric
)

results = evaluator.evaluate(num_samples=10)

print(results)
# summary = summarize(
#     results_df=results,
#     metric_name=metric.name,
#     threshold=0.3
# )

# print("=== Attack Evaluation ===")
# for k, v in summary.items():
#     print(f"{k}: {v:.3f}")
