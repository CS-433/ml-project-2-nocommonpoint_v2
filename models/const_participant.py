from .protocol import BrightenModel

import os
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import pandas as pd


class ConstParticipantModel(BrightenModel):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mapping = None

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        # same... forget about standardizing here, the ranges should be fine
        target_columns = [c for c in df.columns if str(c).startswith("has_")]
        return df.drop(target_columns, axis=1)

    @staticmethod
    def xy_split(df: pd.DataFrame) -> tuple[NDArray, NDArray]:
        y = df["target"].to_numpy()
        x = df["participant_id"].to_numpy()
        return x, y

    def fit(self, x, y, xval=None, yval=None):
        self.mapping = {}
        for i in range(len(x)):
            self.mapping[x[i]] = y[i]

    def predict(self, x):
        array = np.zeros(x.shape[0], dtype=np.int32)
        for i, pid in enumerate(x):
            array[i] = self.mapping[pid]
        return array
