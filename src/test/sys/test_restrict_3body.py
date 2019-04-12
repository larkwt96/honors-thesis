import unittest
import numpy as np
import matplotlib.pyplot as plt
from echonn.sys import SystemSolver, RestrictedCircular3Body


class TestModel(unittest.TestCase):
    def testExample(self):
        sys = RestrictedCircular3Body()
        slv = SystemSolver(sys)
        lce, lce_run = slv.get_lce()
        slv.calc_lce
        # print(lce)
        init = [1, -1, 1, -1]
        run = slv.run([0, 5], init)
        run['results'].y = run['results'].y[:2, :]
        slv.plot2d(run)
        # plt.show(True)

    @unittest.skip
    def testLCE(self):
        # slv.calc_lce(Df_y0, T)
        sys = RestrictedCircular3Body()
        slv = SystemSolver(sys)
        lce, lce_run = slv.get_lce(T=100)
        t = lce_run['results'].t
        y = lce_run['results'].y
        sys_dim = sys.dim
        lces = []
        start_t = 10
        for i in range(start_t, t.shape[0]):
            Df_y0 = y[sys_dim:, i].reshape(sys_dim, sys_dim)
            lce = slv.calc_lce(Df_y0, t[i])
            lces.append(lce)
        plt.plot(t[start_t:], lces)
        plt.show(True)
