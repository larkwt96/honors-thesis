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


class TSData:
    def __init__(self, data=None, run=None, split=0.9):
        """
        data - is a tuple of t and y of the following format:
            t[time point]
            y[dimension, time point]
        run - is what's returned by the system solver
        """
        if run is not None:
            res = run['results']
            data = res.t, res.y
        self.run = run
        self.t, self.y = data
        self.N = self.t.shape[0]
        self.test_index = int(self.N * split)
        self.cv_index = int(self.N * split**2)

        self.train_t = self.t[:self.cv_index]
        self.train_y = self.y[:, :self.cv_index]

        self.validation_t = self.t[self.cv_index:self.test_index]
        self.validation_y = self.y[:, self.cv_index:self.test_index]

        self.test_t = self.t[self.test_index:]
        self.test_y = self.y[:, self.test_index:]

    @property
    def train(self):
        return self.train_t, self.train_y

    @property
    def validation(self):
        return self.validation_t, self.validation_y

    @property
    def test(self):
        return self.test_t, self.test_y
