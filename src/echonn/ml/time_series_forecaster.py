from abc import ABC
import numpy as np


class TimeSeriesForecaster(ABC):
    """
    When it comes to data, here are the naming conventions and formats:
    t = the array of time points (shape: time)
    u = the matrix of inputs (shape: time x dimension)
    d = the target values (shape: time x dimension)
    y = the predicted values (shape: time x dimension)
    """

    def __init__(self, params):
        self.params = params

    @staticmethod
    def rmse(d, y):
        num_samples = d.shape[0]
        return np.sqrt(np.sum((d-y)**2) / num_samples)

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
