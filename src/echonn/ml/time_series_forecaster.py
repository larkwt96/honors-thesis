from abc import ABC
import numpy as np


class TimeSeriesForecaster(ABC):
    @staticmethod
    def rmse(x, y):
        num_samples = x.shape[0]
        return np.sqrt(np.sum((x-y)**2) / num_samples)

    @staticmethod
    def scale_matrix(W):
        if W.shape[1] == 0:
            return W  # no matrix to scale
        return W / np.sqrt(W.shape[1])

    def score(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
