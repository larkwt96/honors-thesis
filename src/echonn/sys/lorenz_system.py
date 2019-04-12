from .system import DynamicalSystem


class LorenzSystem(DynamicalSystem):
    def __init__(self, sigma=10, rho=28, beta=8/3, method='RK45'):
        super().__init__(3, method, 150)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

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
