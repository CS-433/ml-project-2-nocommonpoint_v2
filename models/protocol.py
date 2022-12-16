from typing import Protocol

import pandas as pd
from numpy.typing import NDArray

# Models should provide:
# @static.preprocess(dframe) -> preprocess, just dropping has columns for random forest
# @static.xy_split(dframe) -> return x, y to be passed to fit
# self.fit(x, y) -> fit the model
# self.predict(x) -> predict labels with the model

class BrightenModel(Protocol):
    
    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        ...

    @staticmethod
    def xy_split(df: pd.DataFrame) -> tuple[NDArray, NDArray]:
        ...

    def fit(self, x: NDArray, y: NDArray):
        ...

    def predict(self, x: NDArray) -> NDArray:
        ...

