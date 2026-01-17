import pandas as pd
from xai_bench.datasets.base_dataset import BaseDataset
from pathlib import Path
from typing import Optional


class PrisoneresDataset(BaseDataset):
    def __init__(self, path: Optional[str]=None, **kwargs):
        self.categorical_features = ['race']
        self.numerical_features = ['age', 'sex', 'decile_score', 'priors_count', 'days_in_jail', 'c_days_from_compas', 'is_violent_recid', 'v_decile_score']
        self.target = "is_recid"
        self.task = "classification"
        
        path = str(path) if path is not None else f"{Path(__file__).parent}/compas_recidivism_racial_bias.csv"     
        super().__init__(path, **kwargs)

    def read(self) -> pd.DataFrame:
        self.df_raw = pd.read_csv(self.path, index_col=0)
        return self.df_raw

    def preprocess(self) -> pd.DataFrame:
        assert self.df_raw is not None, "Has to ghave df"
        df = self.df_raw.copy()
        
        self.y_full = df[self.target]
        X = df.drop(columns=[self.target])
        assert self.categorical_features is not None, "Has to ahve them"
        X = self.one_hot_encode_with_mapping(X, self.categorical_features)

        X = X.astype(float)

        for col in X.columns:
            if col not in sum(self.feature_mapping.values(), []):
                self.feature_mapping[col] = [col]

        self.X_full = X.astype(float)
        return self.X_full
    

if __name__ == "__main__":
    path = "src/xai_bench/datasets/compas_recidivism_racial_bias.csv"

    dataset = PrisoneresDataset(path=None)
    assert dataset.df_raw is not None and dataset.X_train is not None and dataset.X_test is not None
    print("Raw data shape:", dataset.df_raw.values.shape)
    print("X_train shape:", dataset.X_train.shape)
    print("X_test shape:", dataset.X_test.shape)

    print("Orignial columns:", dataset.df_raw.columns)
    print("Column mapping:", dataset.feature_mapping)

    

    