import unittest

import matplotlib.pyplot as plt
import numpy as np
from echonn.sys import NBodySystem, SystemSolver
from scipy.constants import G, pi
from .sys_util import clearFigs


class TestModel(unittest.TestCase):
    def setUp(self):
        self.atol = 1e-20

    def runTestPacking(self, mass, dim):
        masses = list(range(1, mass+1))
        system = NBodySystem(body_masses=masses, body_dim=dim)
        vreal = np.arange(2*mass*dim, dtype=np.float64)
        rreal = vreal[:mass*dim].reshape(mass, dim)
        preal = vreal[mass*dim:].reshape(mass, dim)

        r, p = system.unpack(vreal)
        v = system.pack(rreal, preal)
        self.assertTrue(np.all(np.isclose(r, rreal)))
        self.assertTrue(np.all(np.isclose(p, preal)))
        self.assertTrue(np.all(np.isclose(v, vreal)))

    def runTestGradient(self, mass, dim):
        for _ in range(10):
            m = 1+5*np.random.rand(mass)**2
            v = np.random.rand(2*mass*dim)
            system = NBodySystem(m, dim)
            vpsys = system.fun(None, v)
            vptest = self.slowNBodyGradientCalc(v, m, dim)
            close = np.all(np.isclose(vpsys, vptest, atol=self.atol))
            self.assertTrue(close)

    def testAllDims(self):
        for mass in range(2, 6):
            for dim in range(2, 6):
                self.runTestPacking(mass, dim)
                self.runTestGradient(mass, dim)

    def slowNBodyGradientCalc(self, v, masses, body_dim):
        v = np.array(v, dtype=np.float64)
        body_system = NBodySystem(masses, body_dim)
        r, p = body_system.unpack(v)

        # pos derivative
        rp = np.zeros_like(r)
        for i, (m, pos) in enumerate(zip(masses, p)):
            self.assertEqual((len(masses), body_dim), r.shape)
            self.assertEqual((body_dim, ), pos.shape)
            rp[i, :] = pos / m

        pp = np.zeros_like(p)

        for i in range(len(pp)):
            F = -masses[i]*G
            rest = np.zeros_like(r[i])
            for j in range(len(r)):
                if j == i:
                    continue
                diff = r[i] - r[j]
                dist = np.sqrt(np.sum(diff**2))
                rest += masses[j]*diff/dist**3
            pp[i] = F*rest

        return self.pack(rp, pp)

    def pack(self, r, p):
        return np.concatenate((r.reshape(-1), p.reshape(-1)))

    def unpack(self, v, num_mass=-1, dim=-1):
        half = v.shape[0]//2
        return v[:half].reshape(num_mass, dim), v[half:].reshape(num_mass, dim)

    def testBaseCase(self):
        r = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [1, 2, 3],
                      [-1, -1, -1]], dtype=np.float64)
        p = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 1, 1]], dtype=np.float64)
        m = np.array([1, 2, 3, 4], dtype=np.float64)
        v = self.pack(r, p)

        rpreal = np.array([[1, 0, 0],
                           [0, 1/2, 0],
                           [0, 0, 1/3],
                           [1/4, 1/4, 1/4]], dtype=np.float64)

        vp = self.slowNBodyGradientCalc(v, m, 3)
        rp, pp = self.unpack(vp, dim=3)

        system = NBodySystem(m, 3)
        vpsys = system.fun(None, v)
        rpsys, ppsys = self.unpack(vpsys, dim=3)

        self.assertTrue(np.all(np.isclose(rpreal, rp, atol=self.atol)))
        self.assertTrue(np.all(np.isclose(rpreal, rpsys, atol=self.atol)))
        self.assertTrue(np.all(np.isclose(ppsys, pp, atol=self.atol)))
        self.assertTrue(np.all(np.isclose(vp, vpsys, atol=self.atol)))

    def testPlot(self):
        sys = NBodySystem()
        solver = SystemSolver(sys)
        tspan = [0, 100]
        y0 = np.zeros(2*sys.body_dim*len(sys.body_masses))
        y0[0:3] = [0, 0, 0]
        y0[3:6] = [3, 1, 3]
        y0[6:9] = [0, 2, 0]
        run = solver.run(tspan, y0)
        clearFigs()
        solver.plotnd(run)
        # plt.show(True)

    def testOrbit(self):
        Ms = 1.98847 * 10**30  # kg
        # M_earth = 5.9722 * 10**24  # kg
        Mj = 1.899 * 10**27  # kg
        Rj = 778.3 * 10**9  # m
        Tj = 3.743 * 10**8  # s
        # V_earth = 30 * 10**3  # m/s
        # r = 149.6 * 10**9  # m

        vel = 2 * pi * Rj / Tj

        sun_x = np.zeros(3)
        sun_v = np.zeros(3)
        jup_x = np.zeros(3)
        jup_v = np.zeros(3)

        jup_x[0] = Rj
        jup_v[1] = vel
        m = [Ms, Mj]
        v = np.concatenate((sun_x, jup_x, sun_v, jup_v)).reshape(-1)

        sys = NBodySystem(m)
        solver = SystemSolver(sys)
        res = solver.run([0, Tj*1.5], v)
        solver.plotnd(res)
        # plt.show(True)
