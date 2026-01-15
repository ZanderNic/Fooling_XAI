import pandas as pd
from typing import Union
from pathlib import Path
import kagglehub

from xai_bench.datasets.base_dataset import BaseDataset
from sklearn.preprocessing import StandardScaler


class HousingDataset(BaseDataset):
    def __init__(self, path: Union[str, Path] = None, **kwargs):   
        self.categorical_features = [ "waterfront", 'view']
        self.numerical_features = ['bedrooms', 'bathrooms', 'sqft_living','sqft_lot', 'floors', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
        self.target = "price"
        self.task = "regression"
        
        path = str(path) if path is not None else f"{Path(__file__).parent}/housing.csv" 
    
        super().__init__(path, task=self.task, **kwargs)


    def read(self) -> pd.DataFrame:
        self.df_raw = pd.read_csv(self.path)
        return self.df_raw

    def preprocess(self) -> pd.DataFrame:
        assert self.df_raw is not None, "Dataframe was not read in."

        self.df_raw = self._create_month_index(self.df_raw)
        self.df_raw = self.df_raw.drop(columns=["id"])

        df = self.df_raw.copy()

        self.y_full = df[self.target]
        self.y_full = self._scale_target(self.y_full) # scaling target variable
        # inverse transform -> self.scaler_y.inverse_transofrm

        X = df.drop(columns=[self.target])
        X = X.astype(float)

        X = self.one_hot_encode_with_mapping(X, self.categorical_features)

        for col in X.columns:
            if col not in sum(self.feature_mapping.values(), []):
                self.feature_mapping[col] = [col]

        self.X_full = X.astype(float)
        return self.X_full
    

    def _create_month_index(self, df):
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        start = df["date"].min()
        df["month_index"] = ((df["year"] - start.year) * 12 + (df["month"] - start.month))

        df = df.drop(columns=["year", "date", "month"])
        return df
    
    def _scale_target(self, y: pd.Series) -> pd.Series:
        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(
            y.to_numpy().reshape(-1, 1)
        )
        return pd.Series(y_scaled.flatten(), index=y.index)


    


if __name__ == "__main__":
    path = "src/xai_bench/datasets/housing.csv"

    dataset = HousingDataset(path)

    print("Raw data shape:", dataset.df_raw.values.shape)
    print("X_train shape:", dataset.X_train.shape)
    print("X_test shape:", dataset.X_test.shape)

    print("Orignial columns:", dataset.df_raw.columns)
    print("Column mapping:", dataset.feature_mapping)

    