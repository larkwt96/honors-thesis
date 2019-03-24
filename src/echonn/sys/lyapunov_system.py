
"""
This purpose of this class is to calculate the lyapunov exponent for
a dynamical system.

What is needed:
u_t = f^t(x_0) - f^t(x_0) = D_{x_0}f^t(x_0)\dot u_0

so lim t => inf 1/t * ln(D_{x_0} f^t(x_0) * u_0)

f^t(x_0) is the trajectory given initial condition x_0, u_0 is the random perturbation

I need to find
"""
import numpy as np
from .system import DynamicalSystem


class LyapunovSystem(DynamicalSystem):
    def __init__(self, sys):
        super().__init__(dim=sys.dim+sys.dim**2, method=sys.method)
        self.sys = sys

    def build_y0(self, y0):
        I = np.identity(self.sys.dim)
        return np.concatenate((y0, I.reshape(-1)))

    def fun(self, t, v):
        y = v[:self.sys.dim]
        yp = self.sys.fun(t, y)
        phi = v[self.sys.dim:].reshape(self.sys.dim, self.sys.dim)
        Dfun = self.sys.Dfun(t, y)
        phip = Dfun@phi
        return np.concatenate((yp, phip.reshape(-1)))
