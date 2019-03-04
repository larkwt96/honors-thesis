from abc import ABC, abstractmethod


class DynamicalSystem(ABC):
    def __init__(self, dim=1, method='RK45'):
        self.dim = dim
        self._method = method

    @abstractmethod
    def fun(self, t, v):
        return v

    @property
    def method(self):
        return self._method
