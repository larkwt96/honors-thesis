from abc import ABC, abstractmethod
import numpy as np


class DynamicalSystem(ABC):
    def __init__(self, dim=1, method='RK45'):
        self.dim = dim
        self._method = method

    @abstractmethod
    def fun(self, t, v):
        return v

    # TODO get lce T value from empirical results
    # TODO get random IC

    def Dfun(self, t, v):
        ''' This ignores t '''
        eps = np.sqrt(np.finfo(np.float64).eps)
        eps_vs = eps * np.identity(self.dim)
        D = np.zeros_like(eps_vs)
        for i, eps_v in enumerate(eps_vs):
            df = np.array(self.fun(t, v+eps_v), dtype=v.dtype)
            f = self.fun(t, v)
            D[:, i] = (df - f) / eps
        return D

    @property
    def method(self):
        return self._method
