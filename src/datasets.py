import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List


def one_hot_encode_with_mapping(
    df: pd.DataFrame,
    columns: List[str],
    drop_original: bool = True
):
    df = df.copy()
    mapping = {}

    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col)

        mapping[col] = list(dummies.columns)

        df = pd.concat([df, dummies], axis=1)

        if drop_original:
            df.drop(columns=[col], inplace=True)

    return df, mapping


class BaseDataset(ABC):
    def __init__(self, path: str):
        self.path = path
        self.df_raw: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.feature_mapping: Dict[str, List[str]] = {}

    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        pass

    def get_feature_mapping(self) -> Dict[str, List[str]]:
        return self.feature_mapping
    

class HeartDataset(BaseDataset):
    def __init__(self, path: str):
        super().__init__(path)

        self.categorical_features = [
            "cp", "restecg", "slope", "thal"
        ]

        self.target = "target"

    def read(self) -> pd.DataFrame:
        self.df_raw = pd.read_csv(self.path)
        return self.df_raw

    def preprocess(self) -> pd.DataFrame:
        if self.df_raw is None:
            self.read()

        df = self.df_raw.copy()

        y = df[self.target]
        X = df.drop(columns=[self.target])

        X, mapping = one_hot_encode_with_mapping(
            X,
            self.categorical_features
        )

        self.feature_mapping.update(mapping)

        for col in X.columns:
            if col not in sum(self.feature_mapping.values(), []):
                self.feature_mapping[col] = [col]

        self.df = pd.concat([X, y], axis=1)
        return self.df