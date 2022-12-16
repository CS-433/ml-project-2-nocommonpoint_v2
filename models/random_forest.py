from .protocol import BrightenModel

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from numpy.typing import NDArray

class SKLearnRandomForest(BrightenModel):
    def __init__(self, *args, **kwargs):
        self.model = RandomForestClassifier(*args, **kwargs)

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        target_columns = [c for c in df.columns if str(c).startswith('has_')]
        return df.drop(target_columns, axis=1)
       
    @staticmethod
    def xy_split(df: pd.DataFrame) -> tuple[NDArray, NDArray]:
        y = df['target'].to_numpy()
        drop_cols = ['target', 'participant_id', 'date']
        x = df.drop(df.columns.intersection(drop_cols), axis=1) .to_numpy()
        return x, y

    def fit(self, x: NDArray, y: NDArray):
        self.model.fit(x, y)

    def predict(self, x: NDArray) -> NDArray:
        return self.model.predict(x)

