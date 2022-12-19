from .protocol import BrightenModel

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from numpy.typing import NDArray

class SKLearnRandomForest(BrightenModel):
    def __init__(self, *args, existing_forest=None, **kwargs):
        if existing_forest is not None:
            self.model = existing_forest
        else:
            self.model = RandomForestClassifier(*args, **kwargs)

    def fit(self, x: NDArray, y: NDArray):
        self.model.fit(x, y)

    def predict(self, x: NDArray) -> NDArray:
        return self.model.predict(x)

