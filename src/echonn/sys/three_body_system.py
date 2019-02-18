import numpy as np
from .system import DynamicalSystem


class ThreeBodySystem(DynamicalSystem):
    def __init__(self, m1=1, m2=2, m3=3, G=1, use_hamiltonian_def):  # TODO: get G
        super().__init__(18, 'BDF')
        self.m = np.array([m1, m2, m3])
        if use_hamiltonian_def:
            self.fun = self.hamiltonian_fun
        else:
            self.fun = self.redefinition_fun

    def unpack_ham(self, v):
        half = self.dim / 2
        return v[:half].reshape(3, 3), v[half:].reshape(3, 3)

    def pack_ham(self, r, p):
        return np.concatenate((r.reshape(-1), p.reshape(-1)))

    def hamiltonian_fun(self, t, v):
        """
        I can't fully explain this one since I'm no physicist but, it's using
        the idea of Hamiltonian formalism. Basically, the position and
        momentum differential equations can be found from the Hamiltonian
        fairly easily, where the Hamiltonian is usually the total energy
        in the system:
            dr/dt = dH/dp
            dp/dt = -dH/dr
        """
        r, p = self.unpack_ham(v)

        rp = (r.T / self.m).T
        pp = None
        return self.pack_ham(rp, pp)

    def rpp(self, t, v):
        pass

    def redefinition_fun(self, t, v):
        """
        Let y1 = r, y2 = r'. Then, [y1', y2'] = [y2, r'']

        r'' is defined in terms of r1 which is y1.
        """
        pass
