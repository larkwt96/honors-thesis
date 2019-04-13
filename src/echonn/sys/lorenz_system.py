from .system import DynamicalSystem
from .system_solver import SystemSolver
import numpy as np


class LorenzSystem(DynamicalSystem):
    def __init__(self, sigma=10, rho=28, beta=8/3, method='RK45'):
        super().__init__(3, method, 150)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def get_rnd_ic(self, **kwargs):
        T = 30
        y0 = np.random.rand(self.dim)
        slvr = SystemSolver(self)
        run = slvr.run([0, T], y0, **kwargs)
        # tend it towards the attractor
        y0 = run['results'].y[:, -1]
        return y0

    def fun(self, t, v):
        x, y, z = v
        xp = self.sigma * (y - x)
        yp = x * (self.rho - z) - y
        zp = x * y - self.beta * z
        return xp, yp, zp

    def Dfun(self, t, v):
        x, y, z = v
        sigma = self.sigma
        beta = self.beta
        rho = self.rho
        return [
            [-sigma, sigma, 0],
            [rho-z, -1, -x],
            [y, x, -beta],
        ]
