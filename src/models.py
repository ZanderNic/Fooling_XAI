import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class RandomForestModel:
    def __init__(
        self,
        categorical_features: list,
        n_estimators: int = 200,
        random_state: int = 42
    ):
        self.categorical_features = categorical_features

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_features
                )
            ],
            remainder="passthrough"
        )

        self.model = Pipeline(
            steps=[
                ("preprocess", self.preprocessor),
                ("rf", RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state
                ))
            ]
        )

    def fit(self, X: pd.DataFrame, y):
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)
    
    def prediction_margin(self, X: pd.DataFrame):
        proba = self.predict_proba(X)
        return proba.max(axis=1) - proba.min(axis=1)