import pandas as pd
from xai_bench.datasets.base_dataset import BaseDataset


class ForestDataset(BaseDataset):
    def __init__(self, path: str, **kwargs):
        self.categorical_features = [
                                "Wilderness_Area1",
                                "Wilderness_Area2",
                                "Wilderness_Area3",
                                "Wilderness_Area4",
                                "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5",
                                "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10",
                                "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15",
                                "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20",
                                "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25",
                                "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30",
                                "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35",
                                "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"
                                ]

        self.numerical_features = [
                                "Elevation",
                                "Aspect",
                                "Slope",
                                "Horizontal_Distance_To_Hydrology",
                                "Vertical_Distance_To_Hydrology",
                                "Horizontal_Distance_To_Roadways",
                                "Hillshade_9am",
                                "Hillshade_Noon",
                                "Hillshade_3pm",
                                "Horizontal_Distance_To_Fire_Points"
                                ]



        self.target = "Cover_Type"
        self.task = "classification"
        super().__init__(path, **kwargs)

    def read(self) -> pd.DataFrame:
        self.df_raw = pd.read_csv(self.path)
        return self.df_raw

    def preprocess(self) -> pd.DataFrame:
        df = self.df_raw.copy()
        
        self.y_full = df[self.target]
        X = df.drop(columns=[self.target])

        X = self.one_hot_encode_with_mapping(X, self.categorical_features)

        X = X.astype(float)

        for col in X.columns:
            if col not in sum(self.feature_mapping.values(), []):
                self.feature_mapping[col] = [col]

        self.X_full = X.astype(float)
        return self.X_full
    

if __name__ == "__main__":
    path = "src/xai_bench/datasets/covtype_forest.csv"

    #df = pd.read_csv(path)
    



    dataset = ForestDataset(path)

    print("Raw data shape:", dataset.df_raw.values.shape)
    print("X_train shape:", dataset.X_train.shape)
    print("X_test shape:", dataset.X_test.shape)

    print("Orignial columns:", dataset.df_raw.columns)
    print("Column mapping:", dataset.feature_mapping)

    

    