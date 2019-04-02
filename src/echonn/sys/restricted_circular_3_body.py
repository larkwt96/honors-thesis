import numpy as np
from .system import DynamicalSystem


class RestrictedCircular3Body(DynamicalSystem):
    def __init__(self, body_ratio=0.5, method='BDF'):
        super().__init__(4, method)
        if body_ratio < 0 or 1 < body_ratio:
            raise ValueError('body_ratio should be in range [0, 1]')
        self.mu = body_ratio
        self.alpha = 1 - body_ratio

    def r1(self, t, v):
        mu = self.mu
        x1, _, y1, _ = v
        return np.sqrt((x1-mu)**2 + y1**2)

    def r2(self, t, v):
        a = self.alpha
        x1, _, y1, _ = v
        return np.sqrt((x1+a)**2 + y1**2)

    def fun(self, t, v):
        a = self.alpha
        mu = self.mu
        r1 = self.r1(t, v)
        r2 = self.r2(t, v)

        x1, x2, y1, y2 = v
        x1p, y1p = x2, y2
        x2p = 2*y2 + x1 - a/r1**3*(x1-mu) - mu/r2**3*(x1+a)
        y2p = -2*x2 + (1 - a/r1**3 - mu/r2**3)*y1

        return x1p, x2p, y1p, y2p
