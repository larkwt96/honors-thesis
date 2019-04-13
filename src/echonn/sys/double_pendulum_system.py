import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from scipy.constants import g, pi

from .system import DynamicalSystem


class DoublePendulumSystem(DynamicalSystem):
    """
    Equal mass and equal length
    """

    def __init__(self, m1=1, m2=1, l1=1, l2=1):
        super().__init__(4, 'BDF', 100)
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2

        # for wikipedia formula
        #self._theta_pre = 6 / (m*l**2)
        #self._gl = g / l
        #self._moment_pre = -.5*m*l**2

    def get_rnd_ic(self, **kwargs):
        a1 = (0.5 + np.random.rand())*pi  # []
        a2 = 2*np.random.rand()*pi  # [0, 2*pi]
        w1 = np.random.rand() / 100  # [0, .01)
        w2 = np.random.rand() / 100  # [0, .01)
        y0 = np.array([a1, a2, w1, w2])
        return y0

    def fun(self, t, v):
        sin = np.sin
        cos = np.cos
        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        a1, a2, w1, w2 = v

        a1p = w1
        a2p = w2
        w1_num_t1 = sin(a1-a2)*(l1*cos(a1-a2)*w1**2 + w2**2)
        w1_den_t1 = 2*l2*(1 + m1 - cos(a1-a2)**2)
        w1_num_t2 = (1 + 2*m1)*sin(a1) + sin(a1-2*a2)
        w1_den_t2 = l1 * (1 + m1 - cos(a1-a2)**2)
        w1p = w1_num_t1 / w1_den_t1 - w1_num_t2 / w1_den_t2

        w2_num = (1 + m1)*(cos(a1)+l1*w1**2) + cos(a1-a2)*w2**2
        w2_den = 1 + m1 - cos(a1-a2)**2
        w2p = sin(a1 - a2) * w2_num / w2_den
        return a1p, a2p, w1p, w2p

    """ formula from wikipedia
    def fun(self, t, v):
        a1, a2, p1, p2 = v

        # a1p, a2p
        cos_a1a2 = np.cos(a1 - a2)
        theta_den = 16 - 9 * cos_a1a2**2
        factor = self._theta_pre / theta_den
        theta_num1 = 2 * p1 - 3 * cos_a1a2 * p2
        theta_num2 = 8 * p2 - 3 * cos_a1a2 * p1
        a1p = factor * theta_num1
        a2p = factor * theta_num2

        # p1p, p2p
        first_term = a1p * a2p * np.sin(a1-a2)
        p1p = self._moment_pre * (first_term + 3 * self._gl * np.sin(a1))
        p2p = self._moment_pre * (-first_term + self._gl*np.sin(a2))

        return a1p, a2p, p1p, p2p
    """
