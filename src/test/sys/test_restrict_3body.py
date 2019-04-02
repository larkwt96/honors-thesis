import unittest
import numpy as np
import matplotlib.pyplot as plt
from echonn.sys import SystemSolver, RestrictedCircular3Body


class TestModel(unittest.TestCase):
    def testExample(self):
        sys = RestrictedCircular3Body()
        slv = SystemSolver(sys)
        init = [1, -1, 1, -1]
        run = slv.run([0, 5], init)
        run['results'].y = run['results'].y[:2, :]
        slv.plot2d(run)
        # plt.show(True)
