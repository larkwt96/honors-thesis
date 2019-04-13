from abc import ABC, abstractmethod
import numpy as np


class DynamicalSystem(ABC):
    def __init__(self, dim=1, method='RK45', best_lce_T=1):
        self.dim = dim
        self._method = method
        self.best_lce_T = best_lce_T

    @abstractmethod
    def fun(self, t, v):
        return v

    def get_rnd_ic(self, **kwargs):
        return np.random.rand(self.system.dim)

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
