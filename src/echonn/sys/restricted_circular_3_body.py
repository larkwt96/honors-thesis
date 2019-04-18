import numpy as np
from .system import DynamicalSystem
from .system_solver import SystemSolver


class RestrictedCircular3Body(DynamicalSystem):
    def __init__(self, body_ratio=0.5, method='RK45', search=False, max_tries=200, T=200, rmin=0.00000000001, rmax=10):
        super().__init__(4, method, 100)  # TODO: verify this
        if body_ratio < 0 or 1 < body_ratio:
            raise ValueError('body_ratio should be in range [0, 1]')
        self.search = search
        self.max_tries = max_tries
        self.T = T
        self.rmin = rmin
        self.rmax = rmax
        self.mu = body_ratio
        self.alpha = 1 - body_ratio

    def get_rnd_ic(self, verbose=False, **kwargs):
        if not self.search:
            return super().get_rnd_ic()
        max_tries = self.max_tries
        T = self.T
        rmin = self.rmin
        rmax = self.rmax
        if verbose:
            print('Finding good initial condition')
        slvr = SystemSolver(self)
        found = False
        tries = 0
        while not found:
            if verbose:
                print('tries', tries)
            tries += 1
            if tries > max_tries:
                raise Exception('Too many failed tries')
            y0 = np.random.rand(self.dim)
            y0[0] = .5 + y0[0]/2
            y0[1] = 0
            y0[2] = 0
            y0[3] = .5 + y0[3]/2
            run = slvr.run([0, T], y0, **kwargs)
            t = run['results'].t
            y = run['results'].y
            found = True
            for i in range(y.shape[1]):
                r1 = self.r1(t[i], y[:, i])
                r2 = self.r2(t[i], y[:, i])
                if r1 < rmin or r2 < rmin or r1 > rmax or r2 > rmax:
                    progress = int((i+1) / y.shape[1] * 100)
                    fstr = 'failed on {:5} / {:5} ({:3} %) with ({:.2}, {:.2})'
                    if verbose:
                        print(fstr.format(i, y.shape[1], progress, r1, r2))
                    found = False
                    break
        if verbose:
            print('soln', y0)
        return y0

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
