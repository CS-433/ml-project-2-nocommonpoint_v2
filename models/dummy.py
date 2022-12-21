from .protocol import BrightenModel

from typing import Optional

from sklearn.dummy import DummyClassifier

from numpy.typing import NDArray

class MostFrequentPredictor(BrightenModel):
    def __init__(self):
        self.model = DummyClassifier(strategy='prior')

    def fit(self, x: NDArray, y: NDArray, 
            xval: Optional[NDArray] = None, yval: Optional[NDArray] = None):
        self.model.fit(x, y)

    def predict(self, x: NDArray) -> NDArray:
        return self.model.predict(x)

