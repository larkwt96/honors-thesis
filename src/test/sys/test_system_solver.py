import os
import math
import unittest
import numpy as np
from echonn.sys import SystemSolver, CircleSystem, LorenzSystem


class TestModel(unittest.TestCase):
    def setUp(self):
        self.circle_solver = SystemSolver(CircleSystem())
        self.lorenz_solver = SystemSolver(LorenzSystem(10, 28, 8/3))
        self.plt = None

    def tearDown(self):
        if self.plt is not None:
            for fig_num in self.plt.get_fignums():
                self.plt.close(fig_num)

    def testSomething(self):
        res = self.circle_solver.run([0, 7], [0, 1])['res']
        self.assertTrue(res.success)
        for i, t in enumerate(res.t):
            x, y = res.y[:, i]
            x_exp, y_exp = np.array([-np.sin(t), np.cos(t)])
            self.assertAlmostEqual(x, x_exp, places=2)
            self.assertAlmostEqual(y, y_exp, places=2)
        # self.viewRes(res)
        # self.viewCircle()

    def test3dGraph(self):
        res = self.lorenz_solver.run([0, 10], [1, 1, 1])
        self.plt, fig = self.lorenz_solver.plotnd(res, False)
        self.plt, _ = self.lorenz_solver.plot3d(res, False)
        self.assertEqual(4, len(fig.get_axes()))

    def test2dGraph(self):
        res = self.circle_solver.run([0, 10], [1, 1])
        self.plt, fig = self.circle_solver.plotnd(res, False)
        self.plt, _ = self.circle_solver.plot2d(res, False)
        self.assertEqual(3, len(fig.get_axes()))

    def testCircleBox(self):
        t_span = [0, 300]
        res = self.circle_solver.run(t_span, [0, 1])
        self.plt, fig = self.circle_solver.plot2d(res, False)
        ax = fig.get_axes()[0]

        t_dense = np.arange(*t_span, t_span[1]/100)
        box = np.array([np.cos(t_dense), -np.sin(t_dense)])
        ax.plot(*box, '-')
        newbox = np.sqrt(res['res'].y[0, -1]**2 + res['res'].y[1, -1]**2)*box
        ax.plot(*newbox, '-')
        self.plt.show(False)
