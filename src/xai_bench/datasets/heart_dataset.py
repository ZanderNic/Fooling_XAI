import pandas as pd
from typing import Union
from pathlib import Path
import kagglehub

from xai_bench.datasets.base_dataset import BaseDataset


class HeartDataset(BaseDataset):
    def __init__(self, path: Union[str, Path], **kwargs):
        self.categorical_features = [
            "cp", "restecg", "slope", "thal", "sex"
        ]
        self.target = "condition"
        super().__init__(path, **kwargs)

    def read(self) -> pd.DataFrame:
        # if not already downloaded downlaod 
        if self.path.suffix != ".csv":
            raise ValueError("Path must point to csv. (Does not need to exist, but needs to end in .csv)")
        if not self.path.exists():
            path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci","heart_cleveland_upload.csv")
            print(f"Downloaded heart-disease-cleveland-uci dataset to {path}")
            self.path = Path(path)
        self.df_raw = pd.read_csv(self.path)
        return self.df_raw

    def preprocess(self) -> pd.DataFrame:
        assert self.df_raw is not None, "Dataframe was not read in."
        df = self.df_raw.copy()

        self.y_full = df[self.target]
        X = df.drop(columns=[self.target])
        X = X.astype(float)

        X = self.one_hot_encode_with_mapping(X, self.categorical_features)

        for col in X.columns:
            if col not in sum(self.feature_mapping.values(), []):
                self.feature_mapping[col] = [col]

        self.X_full = X.astype(float)
        return self.X_full
    

if __name__ == "__main__":
    import os
    #path = os.path.join("datasets", "heart.csv")
    path = "src/xai_bench/datasets/heart.csv"
    dataset = HeartDataset(path)

    print("Raw data shape:", dataset.df_raw.values.shape)
    print("X_train shape:", dataset.X_train.shape)
    print("X_test shape:", dataset.X_test.shape)

    print("Orignial columns:", dataset.df_raw.columns)
    print("Column mapping:", dataset.feature_mapping)

    