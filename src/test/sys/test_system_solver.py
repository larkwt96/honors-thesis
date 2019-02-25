import unittest
import numpy as np
from .sys_util import clearFigs
import matplotlib.pyplot as plt
from echonn.sys import LorenzSystem
from echonn.sys import CircleSystem
from echonn.sys import SystemSolver


class TestModel(unittest.TestCase):
    def setUp(self):
        # test data
        self.circle_solver = SystemSolver(CircleSystem())
        self.lorenz_solver = SystemSolver(LorenzSystem())

    def testSomething(self):
        res = self.circle_solver.run([0, 7], [0, 1])['results']
        self.assertTrue(res.success)
        for i, t in enumerate(res.t):
            x, y = res.y[:, i]
            x_exp, y_exp = np.array([-np.sin(t), np.cos(t)])
            self.assertAlmostEqual(x, x_exp, places=2)
            self.assertAlmostEqual(y, y_exp, places=2)

    def testCircleBox(self):
        t_span = [0, 300]
        res = self.circle_solver.run(t_span, [0, 1])
        fig = self.circle_solver.plot2d(res, None)
        ax = fig.get_axes()[0]

        t_dense = np.arange(*t_span, t_span[1]/1000)
        box = np.array([np.cos(t_dense), -np.sin(t_dense)])
        ax.plot(*box, '-')
        newbox = np.sqrt(res['results'].y[0, -1]**2 +
                         res['results'].y[1, -1]**2)*box
        ax.plot(*newbox, '-')
        plt.show(False)
        # You should be able to graph things on top of the graphed results
