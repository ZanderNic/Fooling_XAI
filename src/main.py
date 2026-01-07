import pandas as pd
from sklearn.model_selection import train_test_split

from models import RandomForestModel
from explainer import LimeExplainer
from attacks import DistributionShiftAttack, ExplanationAwareDistributionShiftAttack, compute_feature_ranges
from metrics import SpearmanMetric, WassersteinMetric
from attack_evaluator import AttackEvaluator, summarize


df = pd.read_csv("datasets/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestModel(
    categorical_features=["cp", "restecg", "slope", "thal"]
)

rf_model.fit(X_train, y_train)

lime_explainer = LimeExplainer(
    model=rf_model,
    X_train=X_train,
    class_names=["No Disease", "Disease"],
    categorical_features=["cp", "restecg", "slope", "thal"]
    )

feature_ranges = compute_feature_ranges(X_train)

attack = DistributionShiftAttack(
    model=rf_model,
    feature_ranges=feature_ranges,
    epsilon=0.15,
    prob_tolerance=0.02
)

metric = SpearmanMetric(feature_names=list(X.columns))

# attack = ExplanationAwareDistributionShiftAttack(
#     explainer=lime_explainer,
#     metric=metric,
#     model=rf_model,
#     feature_ranges=feature_ranges,
#     epsilon=0.15,
#     prob_tolerance=0.02
# )

evaluator = AttackEvaluator(
    model=rf_model,
    explainer=lime_explainer,
    attack=attack,
    metric=metric
)

results = evaluator.evaluate(X_test.sample(20, random_state=42))

summary = summarize(
    results_df=results,
    metric_name=metric.name,
    threshold=0.3
)

print("=== Attack Evaluation ===")
for k, v in summary.items():
    print(f"{k}: {v:.3f}")