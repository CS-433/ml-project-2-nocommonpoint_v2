"""
Models should provide:
  @static.preprocess(dframe) -> preprocess, just dropping has columns for random forest
  @static.xy_split(dframe) -> return x, y to be passed to fit
  self.fit(x, y) -> fit the model
  self.predict(x) -> predict labels with the model
"""

from typing import Protocol, Optional

import pandas as pd
from numpy.typing import NDArray


class BrightenModel(Protocol):
    """
    A protocol for unifying our models.
    Each model provides its own preprocessing/splitting,
    as well as fit/predict. These are thin wrappers for SKLearn models,
    but somewhat more involved for pytorch.
    """

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        target_columns = [c for c in df.columns if str(c).startswith("has_")]
        return df.drop(target_columns, axis=1)

    @staticmethod
    def xy_split(df: pd.DataFrame) -> tuple[NDArray, NDArray]:
        y = df["target"].to_numpy()
        drop_cols = ["target", "participant_id", "date"]
        x = df.drop(df.columns.intersection(drop_cols), axis=1).to_numpy()
        return x, y

    def fit(
        self, x: NDArray, y: NDArray, xval: Optional[NDArray], yval: Optional[NDArray]
    ):
        ...

    def predict(self, x: NDArray) -> NDArray:
        ...
