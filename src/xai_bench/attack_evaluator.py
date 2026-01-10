import numpy as np
from sklearn.metrics import accuracy_score

from xai_bench.base import BaseDataset, BaseModel, BaseExplainer, BaseAttack, BaseMetric

class AttackEvaluator:
    def __init__(self, dataset: BaseDataset, model:BaseModel, explainer:BaseExplainer, attack: BaseAttack, metric:BaseMetric):
        self.dataset = dataset
        self.model = model
        self.explainer = explainer
        self.attack = attack
        self.metric = metric

    # in order to allow evaluation without always retraining the model
    def fit(self):
        assert self.dataset.X_train is not None and self.dataset.X_test is not None and self.dataset.y_train is not None and self.dataset.y_test is not None, "Dataset not processed"
        self.model.fit(self.dataset.X_train.values, self.dataset.y_train.values)

        acc = accuracy_score(self.dataset.y_test.values, self.model.predict(self.dataset.X_test.values))
        print(f"Accuracy on the test set: {acc:.2f}")

    def evaluate(self, num_samples=1000):
        assert self.dataset.X_train is not None and self.dataset.X_test is not None and self.dataset.y_train is not None and self.dataset.y_test is not None, "Dataset not processed"
        results = []

        if len(self.dataset.X_test) <= num_samples:
            X_test = self.dataset.X_test 
        else:
            X_test = self.dataset.X_test.sample(
                n=num_samples
            )

        for i in range(len(X_test)):
            x = self.dataset.X_test.iloc[i]
            exp = self.explainer.explain(x.values)
            x_adv = self.attack.generate(x.values)
            exp_adv = self.explainer.explain(x_adv)

            results.append({
                self.metric.compute(exp, exp_adv)
            })

        return np.mean(results), np.std(results)

    