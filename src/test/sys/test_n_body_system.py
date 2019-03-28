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

    @unittest.skip
    def testRandomInit(self):
        sys = NBodySystem(body_masses=[1, 1, 1], G=1)
        solver = SystemSolver(sys)
        tspan = [0, 100]
        expand = 10
        y0 = np.random.rand(2*sys.body_dim*len(sys.body_masses))*expand
        y0[9:] /= expand/2

        total_mass = np.sum(sys.body_masses)
        vcm = np.zeros(3)
        for m, v in zip(sys.body_masses, y0[9:].reshape(3, -1)):
            vcm += m*v
        vcm /= total_mass
        yps = y0[9:].reshape(3, -1)
        yps -= vcm[None, :]
        run = solver.run(tspan, y0)
        clearFigs()
        solver.plotnd(run)
        # print(run['results'].y[:, -1].reshape(6, -1))
        y_act = run['results'].y[:9]
        run['results'].y = y_act[:3]
        fig = solver.plot3d(run)
        run['results'].y = y_act[3:6]
        fig = solver.plot3d(run, fig=fig)
        run['results'].y = y_act[6:9]
        fig = solver.plot3d(run, fig=fig)
        plt.show(True)

    @unittest.skip
    def testFig8LcePartition(self):  # takes a long time, so its disabled
        sys = NBodySystem(body_masses=[1, 1, 1], G=1)
        solver = SystemSolver(sys)
        tspan = [0, 10]
        y0 = np.zeros(2*sys.body_dim*len(sys.body_masses), dtype=np.float64)

        x1 = np.array([0.97000436, -0.24308753, 0])
        x3p = np.array([-0.93240737, -0.86473146, 0])

        y0[0:3] = x1
        y0[3:6] = -x1
        y0[6:9] = 0
        y0[9:12] = -x3p / 2
        y0[12:15] = -x3p / 2
        y0[15:18] = x3p
        # print(sys.fun(np.zeros_like(y0), y0).reshape(6, -1))
        tspan, lces = solver.quick_lce(tspan[1], y0, partition=None)
        print(np.mean(lces[-5:]))
        clearFigs()
        plt.figure()
        plt.plot(tspan, lces)
        plt.show(True)

    @unittest.skip
    def testFig8Lce(self):  # takes a long time, so its disabled
        sys = NBodySystem(body_masses=[1, 1, 1], G=1)
        solver = SystemSolver(sys)
        tspan = [0, 1.5]
        y0 = np.zeros(2*sys.body_dim*len(sys.body_masses), dtype=np.float64)

        x1 = np.array([0.97000436, -0.24308753, 0])
        x3p = np.array([-0.93240737, -0.86473146, 0])

        y0[0:3] = x1
        y0[3:6] = -x1
        y0[6:9] = 0
        y0[9:12] = -x3p / 2
        y0[12:15] = -x3p / 2
        y0[15:18] = x3p
        # print(sys.fun(np.zeros_like(y0), y0).reshape(6, -1))
        lce, run = solver.get_lce(tspan[1], y0)
        t = run['results'].t[1:]
        y = run['results'].y[sys.dim:, 1:].reshape(sys.dim, sys.dim, -1)
        #print(y[:, :, -1])
        lces = []
        for i, t_val in enumerate(t):
            Df_y = y[:, :, i]
            lces.append(solver.calc_lce(Df_y, t_val))
        print(lces[-1])
        print(np.mean(lces[-5:]))
        clearFigs()
        plt.figure()
        plt.plot(t, lces)
        plt.show(True)

    def testFig8(self):
        # using IC from TODO
        sys = NBodySystem(body_masses=[1, 1, 1], G=1)
        solver = SystemSolver(sys)
        tspan = [0, 10]
        y0 = np.zeros(2*sys.body_dim*len(sys.body_masses), dtype=np.float64)

        x1 = np.array([0.97000436, -0.24308753, 0])
        x3p = np.array([-0.93240737, -0.86473146, 0])

        y0[0:3] = x1
        y0[3:6] = -x1
        y0[6:9] = 0
        y0[9:12] = -x3p / 2
        y0[12:15] = -x3p / 2
        y0[15:18] = x3p
        # print(sys.fun(np.zeros_like(y0), y0).reshape(6, -1))
        run = solver.run(tspan, y0)
        clearFigs()
        solver.plotnd(run)
        # print(run['results'].y[:, -1].reshape(6, -1))
        y_act = run['results'].y[:9]
        run['results'].y = y_act[:3]
        fig = solver.plot3d(run)
        run['results'].y = y_act[3:6]
        fig = solver.plot3d(run, fig=fig)
        run['results'].y = y_act[6:9]
        fig = solver.plot3d(run, fig=fig)
        # plt.show(True)

    def testTwoPoints(self):
        # using IC from TODO
        sys = NBodySystem(body_masses=[1, 1], G=1)
        solver = SystemSolver(sys)
        max_t = 52
        tspan = [0, max_t]
        y0 = np.zeros(2*sys.body_dim*len(sys.body_masses), dtype=np.float64)
        y0[0:3] = [1, 0, 0]
        y0[3:6] = [-1, 0, 0]
        y0[6:9] = [0, .1, 0]
        y0[9:12] = [0, -.1, 0]
        run = solver.run(tspan, y0)
        # print()
        # print(*y0)
        # print(*run['results'].y[:, -1])
        # clearFigs()
        # fig = solver.plotnd(run)
        # for axes in fig.axes:
        # axes.plot(np.zeros_like(run['results'].y[0]))
        t = run['results'].t
        y = run['results'].y
        # print(t.shape, y.shape)
        # print(t[-1], max_t)
        y0 = y[0]
        y1 = y[3]
        clearFigs()
        plt.figure()
        plt.plot(t, y0)
        plt.plot(t, y1)
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
