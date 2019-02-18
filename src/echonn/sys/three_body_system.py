import numpy as np
from .system import DynamicalSystem
from scipy.constants import G


class ThreeBodySystem(DynamicalSystem):
    def __init__(self, body_masses=[1, 2, 3], body_dim=3):
        body_masses = np.array(body_masses)
        body_count, = body_masses.shape
        dim = 2*body_dim*body_count
        super().__init__(dim, 'BDF')
        # Number of bodies
        self.body_count = body_count
        # Dimension of bodies (3rd or 2nd make the most sense)
        self.body_dim = body_dim
        # The mass of each bodies
        self.body_masses = body_masses

    def unpack_ham(self, v):
        half = self.dim / 2
        r = v[:half].reshape(self.body_count, self.body_dim)
        p = v[half:].reshape(self.body_count, self.body_dim)
        return r, p

    def pack_ham(self, r, p):
        return np.concatenate((r.reshape(-1), p.reshape(-1)))

    def fun(self, t, v):
        r, p = self.unpack_ham(v)
        # Just do the rename variables and differentiate. Make system the
        # Lagrangian and Hamiltonian:
        #     y1 = r (position)
        #     y2 = mr' (momentum)
        rp = r
        pp = p
        return self.pack_ham(rp, pp)
