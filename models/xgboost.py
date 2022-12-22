from .protocol import BrightenModel

import xgboost as xgb
import pandas as pd

from numpy.typing import NDArray


class XGBClassifier(BrightenModel):
    def __init__(self, *args, existing=None, **kwargs):
        if existing is not None:
            self.model = existing
        else:
            self.model = xgb.XGBClassifier(*args, objective="binary:logistic", **kwargs)

    def fit(self, x: NDArray, y: NDArray, xgval=None, yval=None):
        self.model.fit(x, y)

    def predict(self, x: NDArray) -> NDArray:
        return self.model.predict(x)
