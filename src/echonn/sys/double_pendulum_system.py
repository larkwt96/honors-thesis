import numpy as np
from .system import DynamicalSystem


class DoublePendulumSystem(DynamicalSystem):
    """
    Equal mass and equal length
    """

    def __init__(self, m=1, l=1, g=10):  # TODO: get g
        super().__init__(4, 'BDF')
        self.m = m
        self.l = l
        self.g = g

        self._theta_pre = 6 / (m*l**2)
        self._gl = g / l

        self._moment_pre = -.5*m*l**2

    def fun(self, t, v):
        a1, a2, p1, p2 = v
        g = self.g

        # a1p, a2p
        cosa1a2 = np.cos(a1 - a2)
        theta_den = 16 - 9 * cosa1a2**2
        factor = self._theta_pre / theta_den
        theta_num1 = 2 * p1 - 3 * cosa1a2 * p2
        theta_num2 = 8 * p2 - 3 * cosa1a2 * p1
        a1p = factor * theta_num1
        a2p = factor * theta_num2

        # p1p, p2p
        first_term = a1p * a2p * np.sin(a1-a2)
        p1p = self._moment_pre * (first_term + 3 * self._gl * np.sin(a1))
        p2p = self._moment_pre * (-first_term + self._gl*np.sin(a2))

        return a1p, a2p, p1p, p2p
