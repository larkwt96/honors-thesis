import unittest
import numpy as np
from echonn.sys import LorenzSystem, DoublePendulumSystem, SystemSolver, CircleSystem
import matplotlib.pyplot as plt
from scipy.constants import pi
from .sys_util import clearFigs


class TestLCECalc(unittest.TestCase):
    def runLCETest(self, sigma, rho, beta, l1):
        sys = LorenzSystem(sigma, rho, beta)
        slv = SystemSolver(sys)
        lce, run = slv.get_lce(T=100)
        T0 = 0
        t = run['results'].t[T0:]
        y = run['results'].y[:, T0:]
        lces = []
        for i, t_val in enumerate(t):
            Df_y0 = y[sys.dim:, i].reshape(sys.dim, sys.dim)
            lces.append(slv.calc_lce(Df_y0, t_val))
        clearFigs()
        plt.figure()
        plt.plot(t, lces)
        plt.show(True)
        print(Df_y0)
        print(lce, l1, (lce - l1)/l1)
        self.assertAlmostEqual((lce - l1)/l1, 0, places=0)

    def runLCETestShort(self, sigma, rho, beta, l1):
        sys = LorenzSystem(sigma, rho, beta)
        slv = SystemSolver(sys)
        slv.get_lce(T=2)

    def setUp(self):
        self.data = [
            [16, 45.92, 4, 1.50255],
            [16, 40, 4, 1.37446],
            [10, 28, 8/3, .90566],
        ]

    def testLCEShort(self):
        for args in self.data:
            self.runLCETestShort(*args)

    @unittest.skip
    def testLCE(self):
        for args in self.data:
            self.runLCETest(*args)

    def testLCEPendulum(self):
        sys = DoublePendulumSystem()
        slv = SystemSolver(sys)
        slv.get_lce(T=2, y0=[1.8, 1.8, 0, 0])

    def testCircleLCE(self):
        sys = SystemSolver(CircleSystem())
        lce, run = sys.get_lce()
        self.assertAlmostEqual(lce, 0, places=4)
        # print(lce)

    @unittest.skip
    def testLCEPendulumLong(self):
        sys = DoublePendulumSystem()
        slv = SystemSolver(sys)
        theta = pi * 120 / 180
        lce, run = slv.get_lce(T=100, y0=[theta, 0, 0, 0])
        y = run['results'].y
        Df_y0 = y[sys.dim:, -1].reshape(sys.dim, sys.dim)
        print(Df_y0)
        print(lce)
        self.assertGreater(lce, 0)

    @unittest.skip
    # cool experiment. Chaos correlates with energy in the system. Also takes 10 minutes to run..
    def testLCEPendulumVaryAngle(self):
        sys = DoublePendulumSystem()
        slv = SystemSolver(sys)
        thetas = []
        lces = []
        for theta in np.arange(0.001, pi+0.0001, pi/20):
            lce, _ = slv.get_lce(T=100, y0=[theta, 0, 0, 0])
            thetas.append(theta)
            lces.append(lce)
        clearFigs()
        plt.figure()
        plt.plot(thetas, lces)
        plt.show(True)
