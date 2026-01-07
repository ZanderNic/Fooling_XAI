import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List
from sklearn.model_selection import train_test_split


class BaseDataset(ABC):
    def __init__(
        self,
        path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ):
        self.path = path
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

        self.df_raw: pd.DataFrame | None = None
        self.X_full: pd.DataFrame | None = None
        self.y_full: pd.Series | None = None

        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

        self.feature_mapping: Dict[str, List[str]] = {}

        self._load_and_prepare()

    def _load_and_prepare(self):
        self.read()
        self.preprocess()
        self._split()

    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        pass

    def _split(self):
        X = self.X_full
        y = self.y_full

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None
        )

    def one_hot_encode_with_mapping(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drop_original: bool = True
    ) -> pd.DataFrame:
        df = df.copy()
        mapping = {}

        for col in columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            mapping[col] = list(dummies.columns)
            df = pd.concat([df, dummies], axis=1)
            if drop_original:
                df.drop(columns=[col], inplace=True)

        self.feature_mapping.update(mapping)
        return df

    def explanation_to_array(self, explanation, feature_order=None):
        feature_order = feature_order or list(self.feature_mapping.keys())

        # For Lime
        if hasattr(explanation, "as_list"):
            exp_dict = dict(explanation.as_list())
        # For Shap
        elif hasattr(explanation, "values") and hasattr(explanation, "feature_names"):
            exp_dict = dict(zip(explanation.feature_names, explanation.values))
        else:
            raise ValueError("Unsupported explanation type")

        arr = []
        for feat in feature_order:
            sub_features = self.feature_mapping.get(feat, [feat])
            importance = sum(exp_dict.get(sf, 0.0) for sf in sub_features)
            arr.append(importance)

        return np.array(arr)

    def get_feature_mapping(self) -> Dict[str, List[str]]:
        return self.feature_mapping