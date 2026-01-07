import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

class LimeExplainer:
    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        class_names=None,
        categorical_features=None,
        discretize_continuous=True
        ):
        self.model = model
        self.X_train = X_train
        self.feature_names = list(X_train.columns)
        self.num_features = len(self.feature_names)

        self.class_names = class_names or ["class_0", "class_1"]

        self.categorical_features = (
            [self.feature_names.index(f) for f in categorical_features]
            if categorical_features else None
        )

        self.explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=self.categorical_features,
            mode="classification",
            discretize_continuous=discretize_continuous
        )

    def _predict_proba(self, x_np):
        x_df = pd.DataFrame(x_np, columns=self.feature_names)
        return self.model.predict_proba(x_df)
    
    def explain(self, x: pd.Series):
        if isinstance(x, pd.Series):
            x = x.values

        exp = self.explainer.explain_instance(
            data_row=x,
            predict_fn=self._predict_proba,
            num_features=self.num_features
        )

        return exp