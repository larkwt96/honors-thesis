import numpy as np
from .system import DynamicalSystem
from scipy.constants import G
from sklearn.metrics.pairwise import euclidean_distances


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

    def diag_multi(self, v, A):
        """
        v is a vector
        A is a matrix

        Returns diag(v)*A
        """
        return (v*A.T).T

    def get_dist_matrix(self, r):
        pass

    def total_acc(self, r):
        m = self.body_masses
        total = []
        dist_matrix = euclidean_distances(r)  # cdist is more accurate but slow
        for i in range(self.body_count):
            total.append([])
            for j in range(i + 1, self.body_count):
                mass = m[i]*m[j]
                diff = r[i] - r[j]
                dist = np.sqrt(diff**2).sum()
                total[i].append(mass * diff / dist**3)
        return total

    def fun(self, t, v):
        # Just do the rename variables and differentiate thing. Make system the
        # Lagrangian and Hamiltonian:
        #     y1 = r (position)
        #     y2 = mr' (momentum)
        #
        # r' = diag(m) P
        # p' = -G diag(m) sum j!=i of m_j (r_i - r_j) / |r_i - r_j|^3
        #  where r_i is the ith row of the R matrix
        #
        # TODO: make this more efficient
        r, p = self.unpack_ham(v)
        m = self.body_masses

        # position derivative
        rp = self.diag_multi(1/m, p)

        # momentum derivative
        diff = r[:, None, :] - r  # broadcast each row to each other
        dist = np.sqrt((diff**2).sum(axis=2))  # l2 norm of each difference

        # divide diff by dist but replace div by zero with 0
        dist_is_not_zero = dist[:, :, None] != 0
        acc = np.divide(diff, dist[:, :, None],
                        out=np.zeros_like(diff),
                        where=dist_is_not_zero)

        # apply rest of gravity equation
        pp = -G * m[None, :] @ acc @ m[:, None]
        return self.pack_ham(rp, pp)
