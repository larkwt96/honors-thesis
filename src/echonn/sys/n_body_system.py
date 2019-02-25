import numpy as np
from .system import DynamicalSystem
from scipy.constants import G


class NBodySystem(DynamicalSystem):
    def __init__(self, body_masses=[1, 2, 3], body_dim=3):
        """
        Warning: changing the body_dim from 3 will work, but the physics
        probably isn't right since it will use the squared distance instead
        of what a 4th dimension gravity equation would use (cubed distance?).
        """
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

    def unpack(self, v):
        half = self.dim // 2
        r = v[:half].reshape(self.body_count, self.body_dim)
        p = v[half:].reshape(self.body_count, self.body_dim)
        return r, p

    def pack(self, r, p):
        return np.concatenate((r.reshape(-1), p.reshape(-1)))

    def fun(self, t, v):
        """
        make sure v is of type np.float64
        """
        # Just do the rename variables and differentiate thing. Make system the
        # Lagrangian and Hamiltonian:
        #     y1 = r (position)
        #     y2 = mr' (momentum)
        #
        # r' = diag(m) P
        # p' = -G diag(m) sum j!=i of m_j (r_i - r_j) / |r_i - r_j|^3

        #  where r_i is the ith row of the R matrix
        r, p = self.unpack(v)
        m = self.body_masses

        # position derivative
        rp = p / m[:, None]

        # momentum derivative
        diff = r[:, None, :] - r  # broadcast each row to each other
        dist = np.sqrt((diff**2).sum(axis=2))**3  # l2 norm of each difference
        # divide diff by dist but replace div by zero with 0
        acc = np.divide(diff, dist[:, :, None],
                        out=np.zeros_like(diff),
                        where=dist[:, :, None] != 0)

        # apply rest of gravity equation
        mass_gravity = (-G*m[:, None]*m)[:, :, None]
        pp = (mass_gravity * acc).sum(axis=1)

        # pack and return
        return self.pack(rp, pp)
