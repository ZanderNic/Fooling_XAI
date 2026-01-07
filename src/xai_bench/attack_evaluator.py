import numpy as np
from sklearn.metrics import accuracy_score


class AttackEvaluator:
    def __init__(self, dataset, model, explainer, attack, metric):
        self.dataset = dataset
        self.model = model
        self.explainer = explainer
        self.attack = attack
        self.metric = metric

    def evaluate(self, num_samples=1000):
        self.model.fit(self.dataset.X_train.values, self.dataset.y_train.values)

        acc = accuracy_score(self.dataset.y_test.values, self.model.model.predict(self.dataset.X_test.values))
        print(f"Accuracy on the test set: {acc:.2f}")

        results = []

        if len(self.dataset.X_test) <= num_samples:
            X_test = self.dataset.X_test 
        else:
            X_test = self.dataset.X_test.sample(
                n=num_samples
            )

        for i in range(len(X_test)):
            x = self.dataset.X_test.iloc[i]
            exp = self.explainer.explain(x)
            x_adv = self.attack.generate(x)
            exp_adv = self.explainer.explain(x_adv)

            results.append({
                self.metric.compute(exp, exp_adv)
            })

        return np.mean(results), np.std(results)

    