from .system import DynamicalSystem


class LorenzSystem(DynamicalSystem):
    def __init__(self, sigma, rho, beta):
        super().__init__(3, 'BDF')
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def fun(self, t, v):
        x, y, z = v
        xp = self.sigma * (y - x)
        yp = x * (self.rho - z) - y
        zp = x * y - self.beta * z
        return xp, yp, zp
