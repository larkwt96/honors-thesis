import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import G as gravity_acc

from .system import DynamicalSystem


class NBodySystem(DynamicalSystem):
    def __init__(self, body_masses=[1, 1.1, 1.2], body_dim=3, G=gravity_acc):
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
        self._body_dim = body_dim
        # The mass of each bodies
        self._body_masses = body_masses
        self.G = G

    @property
    def body_dim(self):
        return self._body_dim

    @property
    def body_masses(self):
        return self._body_masses

    def unpack(self, v):
        if v.dtype != np.float64:
            v = v.astype(np.float64)
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
        G = self.G
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

    def render_fade_trail2d(self, run, fig=None):
        # load figure
        if fig is None:
            fig = plt.figure()
        else:
            plt.figure(fig.number)

    def render_fade_trail3d(self, run, fig=None):
        # load figure
        if fig is None:
            fig = plt.figure()
        else:
            plt.figure(fig.number)

    def render_time_frame(self, t, y, fig, time=1):
        pass

    def render_fade_trail(self, run, fig=None, base_size=10):
        # select dim
        res = run['results']
        sys = run['system']
        if sys.dim == 1:
            msg = "while possible, can't yet"
            raise NotImplementedError(msg)
        elif sys.dim == 2:
            self.render_fade_trail2d(run, fig)
        elif sys.dim == 3:
            self.render_fade_trail3d(run, fig)
        else:
            raise Exception("Can't render {}th dimension")

        # get data from run
        t = res.t
        y = res.y
        dim = sys.body_dim

        """
        tmask = np.where((time - 1 <= t) & (t <= time))
        body_pos = y[:dim//2, tmask].reshape(-1, dim)
        r, p = sys.unpack(y)
        """
