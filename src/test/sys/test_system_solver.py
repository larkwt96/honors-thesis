import os
import math
import unittest
import numpy as np
from echonn.sys import SystemSolver, CircleSystem, LorenzSystem


class TestModel(unittest.TestCase):
    def setUp(self):
        self.circle_solver = SystemSolver(CircleSystem())
        self.lorenz_solver = SystemSolver(LorenzSystem())
        self.plt = None  # used to clear graphs
        self.clear = False  # clear figures after every test

    def tearDown(self):
        if self.clear and self.plt is not None:
            for fig_num in self.plt.get_fignums():
                self.plt.close(fig_num)

    def testSomething(self):
        res = self.circle_solver.run([0, 7], [0, 1])['results']
        self.assertTrue(res.success)
        for i, t in enumerate(res.t):
            x, y = res.y[:, i]
            x_exp, y_exp = np.array([-np.sin(t), np.cos(t)])
            self.assertAlmostEqual(x, x_exp, places=2)
            self.assertAlmostEqual(y, y_exp, places=2)

    def test3dGraph(self):
        res = self.lorenz_solver.run([0, 10], [1, 1, 1])
        self.plt, fig = self.lorenz_solver.plotnd(res, block=False)
        self.plt, _ = self.lorenz_solver.plot3d(res, block=False)
        self.assertEqual(4, len(fig.get_axes()))

    def test2dGraph(self):
        res = self.circle_solver.run([0, 10], [1, 1])
        self.plt, fig = self.circle_solver.plotnd(res, block=False)
        self.plt, _ = self.circle_solver.plot2d(res, block=False)
        self.assertEqual(3, len(fig.get_axes()))

    def testCircleBox(self):
        t_span = [0, 300]
        res = self.circle_solver.run(t_span, [0, 1])
        self.plt, fig = self.circle_solver.plot2d(res, None, False)
        ax = fig.get_axes()[0]

        t_dense = np.arange(*t_span, t_span[1]/1000)
        box = np.array([np.cos(t_dense), -np.sin(t_dense)])
        ax.plot(*box, '-')
        newbox = np.sqrt(res['results'].y[0, -1]**2 +
                         res['results'].y[1, -1]**2)*box
        ax.plot(*newbox, '-')
        self.plt.show(False)
        # You should be able to graph things on top of the graphed results

    def testMultiGraph(self):
        run1 = self.lorenz_solver.run([0, 20], [1, 1, 1])
        run2 = self.lorenz_solver.run([0, 20], [1, 1, 1])
        run3 = self.lorenz_solver.run([0, 20], [1, 1, 1+10**-9])
        _, fig = self.lorenz_solver.plot3d(run1, block=False)
        _, fig = self.lorenz_solver.plot3d(run2, fig, False)
        self.lorenz_solver.plot3d(run3, fig, False)
        # You should see that orange (2nd graph) covers blue (1st graph) while
        # adding a billionth to green (3rd graph) causes it to diverge.

    def testMulti2dGraph(self):
        run1 = self.circle_solver.run([0, 20], [0, 2])
        run2 = self.circle_solver.run([0, 20], [0, 1])
        _, fig = self.circle_solver.plot2d(run1, block=False)
        self.circle_solver.plot2d(run2, fig, False)

    def testMultiNdGraph(self):
        run1 = self.lorenz_solver.run([0, 20], [1, 1, 1])
        run2 = self.lorenz_solver.run([0, 20], [1, 1, 1.001])
        _, fig = self.lorenz_solver.plotnd(run1, block=False)
        self.lorenz_solver.plotnd(run2, fig, block=False)
