import unittest
import numpy as np
from echonn.sys import LorenzSystem, DoublePendulumSystem, SystemSolver
import matplotlib.pyplot as plt


class TestLCECalc(unittest.TestCase):
    def setUp(self):
        self.validation_test = False

    def runLCETest(self, sigma, rho, beta, l1):
        sys = LorenzSystem(sigma, rho, beta)
        slv = SystemSolver(sys)
        slv.get_lce(T=2)
        # uncomment to validate
        if self.validation_test:
            lce, run = slv.get_lce(T=100)
            T0 = 0
            t = run['results'].t[T0:]
            y = run['results'].y[:, T0:]
            lces = []
            for i, t_val in enumerate(t):
                Df_y0 = y[sys.dim:, i].reshape(sys.dim, sys.dim)
                lces.append(slv.calc_lce(Df_y0, t_val))
            plt.figure()
            plt.plot(t, lces)
            plt.show(True)
            print(Df_y0)
            print(lce, l1, (lce - l1)/l1)
            self.assertAlmostEqual((lce - l1)/l1, 0, places=0)

    def testLCE(self):
        data = [
            [16, 45.92, 4, 1.50255],
            [16, 40, 4, 1.37446],
            [10, 28, 8/3, .90566],
        ]
        for args in data:
            self.runLCETest(*args)

    def testLCEPendulum(self):
        sys = DoublePendulumSystem()
        slv = SystemSolver(sys)
        slv.get_lce(T=2, y0=[1.8, 1.8, 0, 0])
        if self.validation_test:
            lce, run = slv.get_lce(T=100, y0=[1.8, 1.8, 0, 0])
            y = run['results'].y
            Df_y0 = y[sys.dim:, -1].reshape(sys.dim, sys.dim)
            print(Df_y0)
            print(lce)
            self.assertGreater(lce, 0)
