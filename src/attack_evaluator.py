import pandas as pd
from tqdm import tqdm

class AttackEvaluator:

    def __init__(self, model, explainer, attack, metric):
        self.model = model
        self.explainer = explainer
        self.attack = attack
        self.metric = metric

    def evaluate(self, X_test: pd.DataFrame):
        results = []

        for i in tqdm(range(len(X_test))):
            x = X_test.iloc[i]

            y = self.model.predict(pd.DataFrame([x]))[0]
            exp = self.explainer.explain(x)

            x_adv = self.attack.generate(x)

            y_adv = self.model.predict(pd.DataFrame([x_adv]))[0]
            exp_adv = self.explainer.explain(x_adv)

            results.append({
                "prediction_same": y == y_adv,
                self.metric.name: self.metric.compute(exp, exp_adv)
            })

        return pd.DataFrame(results)
    
def summarize(results_df, metric_name, threshold=0.3):
    return {
        "ASR": (
            (results_df["prediction_same"]) &
            (results_df[metric_name] > threshold)
        ).mean(),
        "mean_distance": results_df[metric_name].mean(),
        "std_distance": results_df[metric_name].std()
    }