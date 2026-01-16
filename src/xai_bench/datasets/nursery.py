import pandas as pd
from typing import Union
from pathlib import Path
import kagglehub

from xai_bench.datasets.base_dataset import BaseDataset


class NurseryDataset(BaseDataset):
    def __init__(self, path: Union[str, Path] = None, **kwargs):   
        self.categorical_features = ['parents', 'has_nurs', 'form', 'housing', 'finance','social', 'health', 'children']
        self.numerical_features = []
        self.target = "final evaluation"
        self.task = "classification"
            
        path = str(path) if path is not None else f"{Path(__file__).parent}/nursery.csv" 

        self.target_mapping = {
                            "not_recom": 0,
                            "priority": 1,
                            "spec_prior": 2,
                            "very_recom": 3,
                            "recommend": 4
                            }
        
        self.inverse_target_mapping_dict = {
            v: k for k, v in self.target_mapping.items()
        }

        super().__init__(path, task=self.task, **kwargs)


    def read(self) -> pd.DataFrame:
        self.df_raw = pd.read_csv(self.path)
        return self.df_raw

    def preprocess(self) -> pd.DataFrame:
        assert self.df_raw is not None, "Dataframe was not read in."

        df = self.df_raw.copy()

        self.y_full = self._encode_target(df[self.target])

        X = df.drop(columns=[self.target])
        X = self.one_hot_encode_with_mapping(X, self.categorical_features)

        self.X_full = X.astype(float)
        return self.X_full
    
    def _encode_target(self, y: pd.Series) -> pd.Series:
        y_encoded = y.map(self.target_mapping)
        if y_encoded.isna().any():
            unknown = y[y_encoded.isna()].unique()
            raise ValueError(f"Unknown target labels: {unknown}")
        
        return y_encoded.astype(int)
    
    def decode_target(self, y: pd.Series) -> pd.Series:
        return y.map(self.inverse_target_mapping_dict)




if __name__ == "__main__":
    import os
    path = "src/xai_bench/datasets/nursery.csv"

    dataset = NurseryDataset(path)


    # print("Raw data shape:", dataset.df_raw.values.shape)
    # print("X_train shape:", dataset.X_train.shape)
    # print("X_test shape:", dataset.X_test.shape)

    # print("Orignial columns:", dataset.df_raw.columns)
    # print("Column mapping:", dataset.feature_mapping)

    # print(dataset.X_full)
    
    # encoding labels 
    pred = dataset.y_full
    