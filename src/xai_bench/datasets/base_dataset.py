import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Literal, cast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xai_bench.explainer.base_explainer import Features, Explanation
from pathlib import Path

class BaseDataset(ABC):
    def __init__(
        self,
        path: Union[str,Path]="",
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        task : Optional[Literal["classification","regression"]] = None
    ):
        self.path = Path(path)
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        if self.task is None:
            self.task: Optional[Literal["classification","regression"]]= task

        self.classes: Optional[list] = None
        self.num_classes: Optional[int] = None
        self.df_raw: Optional[pd.DataFrame] = None
        self.X_full: Optional[pd.DataFrame]= None
        self.y_full: Optional[pd.Series] = None

        self.X_train: Optional[pd.DataFrame]= None
        self.X_test: Optional[pd.DataFrame]= None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.features: Optional[Features] = None
        self.feature_mapping: Dict[str, List[str]] = {}
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.y_range: dict[str,float] = {}

        self.categorical_features: Optional[List[str]]
        self.numerical_features: Optional[List[str]]

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
        assert self.X_full is not None and self.y_full is not None, "Dataset not processed."
        X = self.X_full
        y = self.y_full

        if self.task == 'classification':
            self.classes = cast(list, y.unique().tolist())
            self.num_classes = len(self.classes)
            stratify_target = y if self.stratify else None
        else: # regression
            self.classes = []
            self.num_classes = 1
            stratify_target = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_target
        )

        # scale numerical features
        assert self.numerical_features is not None, "Has to have num features"
        assert self.X_train is not None and self.X_test is not None, "has to have xtrain"
        if len(self.numerical_features) > 0:

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train[self.numerical_features]) # type: ignore
            X_test_scaled = self.scaler.transform(self.X_test[self.numerical_features]) # type: ignore
            # traindata scaled
            self.X_train_scaled = pd.concat([
                pd.DataFrame(
                    X_train_scaled,
                    columns=self.numerical_features, 
                    index=self.X_train.index
                ),
                self.X_train.drop(self.numerical_features, axis=1)
            ], axis=1)

            # testdata scaled
            self.X_test_scaled = pd.concat([
                pd.DataFrame(
                    X_test_scaled,
                    columns=self.numerical_features,
                    index=self.X_test.index
                ),
                self.X_test.drop(self.numerical_features, axis=1)
            ], axis=1)
        else:
            # no numerical features
            self.scaler = None
            self.X_test_scaled = self.X_test.copy()
            self.X_train_scaled = self.X_train.copy()



        self.features = Features(list(self.X_full.columns))
        
        # determine scaled and unscaled feature ranges
        self.feature_ranges = {
            col: (self.X_train[col].min(), self.X_train[col].max()) for col in self.X_train.columns
        }
        self.scaled_feature_ranges = {
            col: (self.X_train_scaled[col].min(), self.X_train_scaled[col].max()) for col in self.X_train.columns
        }

        # determine scaled and unscaled categorical values
        self.categorical_values = {
            col: self.X_train[col].unique() if col not in self.numerical_features else None
            for col in self.X_train.columns
        }
        self.scaled_categorical_values = {
            col: self.X_train_scaled[col].unique() if col not in self.numerical_features else None
            for col in self.X_train.columns
        }

        # produce a list mask for categorical features
        self.categorical_feature_mask = [
            True if feature not in self.numerical_features else 
            False 
            for feature in self.features.feature_names_model
        ]


    def one_hot_encode_with_mapping(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drop_original: bool = True
    ) -> pd.DataFrame:
        df = df.copy()
        mapping = {}

        for col in columns:
            dummies = pd.get_dummies(df[col], prefix=col).astype(float)
            mapping[col] = list(dummies.columns)
            df = pd.concat([df, dummies], axis=1)
            if drop_original:
                df.drop(columns=[col], inplace=True)

        self.feature_mapping.update(mapping)
        return df

    def explanation_to_array(
        self,
        explanation: Explanation,
        feature_order=None
    ) -> np.ndarray:
        feature_order = feature_order or list(self.feature_mapping.keys())

        # extract explanation values into a dict
        if isinstance(explanation, Explanation) or \
            (hasattr(explanation, 'values') and hasattr(explanation, 'feature_names')):
            exp_dict = dict(zip(explanation.feature_names, explanation.values.T))
        else:
            raise ValueError("Unsupported explanation type")

        arr = []
        for feat in feature_order:
            sub_features = self.feature_mapping.get(feat, [feat])
            importance = sum(exp_dict.get(sf, 0.0) for sf in sub_features)
            arr.append(importance)

        return np.array(arr).T

    def get_feature_mapping(self) -> Dict[str, List[str]]:
        return self.feature_mapping
    
    def __str__(self) -> str:
        if self.X_full is not None:
            return f"Dataset(name={self.__class__.__name__}, samples={len(self.X_full)}, features={len(self.X_full.columns)})"
        else:
            return f"Dataset(name={self.__class__.__name__}, not yet loaded)"