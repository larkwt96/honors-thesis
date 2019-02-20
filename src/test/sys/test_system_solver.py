import os
import time
import math
import unittest
import numpy as np
from scipy.constants import G
from echonn.sys import SystemSolver
from echonn.sys import CircleSystem
from echonn.sys import LorenzSystem
from echonn.sys import DoublePendulumSystem
from echonn.sys import NBodySystem
import matplotlib.pyplot as plt


class TestModel(unittest.TestCase):

    def setUp(self):
        # test data
        self.circle_solver = SystemSolver(CircleSystem())
        self.lorenz_solver = SystemSolver(LorenzSystem())
        self.pendulum_system = DoublePendulumSystem()
        self.pendulum_solver = SystemSolver(self.pendulum_system)
        self.n_body_system = NBodySystem()
        self.n_body_solver = SystemSolver(self.n_body_system)
        self.n_body_data = []
        self.n_body_expected = []

    def clearFigs(self):
        # TODO: it doesn't want to close figures, and I don't know why. In the
        # mean time, I will just clear the figure, and the non blank figure is
        # the one you want.
        for fignum in plt.get_fignums():
            plt.figure(fignum)
            plt.clf()
        """
        while len(plt.get_fignums()) > 0:
            fignum = plt.get_fignums()[0]
            plt.close(fignum)
            plt.pause(.001) # TODO: try this?
        """

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
        fig = self.lorenz_solver.plotnd(res)
        self.lorenz_solver.plot3d(res)
        self.assertEqual(4, len(fig.get_axes()))

    def test2dGraph(self):
        res = self.circle_solver.run([0, 10], [1, 1])
        fig = self.circle_solver.plotnd(res)
        self.circle_solver.plot2d(res)
        self.assertEqual(3, len(fig.get_axes()))

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

    def testMultiGraph(self):
        run1 = self.lorenz_solver.run([0, 20], [1, 1, 1])
        run2 = self.lorenz_solver.run([0, 20], [1, 1, 1])
        run3 = self.lorenz_solver.run([0, 20], [1, 1, 1+10**-9])
        fig = self.lorenz_solver.plot3d(run1)
        fig = self.lorenz_solver.plot3d(run2, fig)
        self.lorenz_solver.plot3d(run3, fig)
        # You should see that orange (2nd graph) covers blue (1st graph) while
        # adding a billionth to green (3rd graph) causes it to diverge.

    def testMulti2dGraph(self):
        run1 = self.circle_solver.run([0, 20], [0, 2])
        run2 = self.circle_solver.run([0, 20], [0, 1])
        fig = self.circle_solver.plot2d(run1)
        self.circle_solver.plot2d(run2, fig)

    def testMultiNdGraph(self):
        run1 = self.lorenz_solver.run([0, 20], [1, 1, 1])
        run2 = self.lorenz_solver.run([0, 20], [1, 1, 1.001])
        fig = self.lorenz_solver.plotnd(run1)
        self.lorenz_solver.plotnd(run2, fig)

    def testDoublePendulum(self):
        self.clearFigs()
        run = self.pendulum_solver.run([0, 2], [.2, 1, 0, 0])
        run2 = self.pendulum_solver.run([0, 2], [.2, 1.1, 0, 0])
        fig = self.pendulum_solver.plotnd(run)
        self.pendulum_solver.plotnd(run2, fig)
        # plt.show(True)

        self.clearFigs()
        system = DoublePendulumSystem()
        fig = system.render_path(run, dot_size=2)
        system.render_path(run2, fig=fig, dot_size=2)
        # plt.show(True)

    def testDoublePendulumFade(self):
        self.clearFigs()
        run = self.pendulum_solver.run([0, 5], [.2, 1, 0, 0])
        system = DoublePendulumSystem()
        if False:
            fig = system.render_fade_trail(run)
            plt.show(False)
            step = .02
            for t in np.arange(.5, 5, step)[1:]:
                plt.pause(.001)
                plt.clf()
                fig = system.render_fade_trail(run, fig=fig, time=t)

    def testDoublePendulumTrail(self):
        self.clearFigs()
        run = self.pendulum_solver.run([0, 10], [.2, 1, 0, 0])
        system = DoublePendulumSystem()
        if False:
            fig = system.render_trail(run, time=.5)
            plt.show(False)
            step = .05
            for t in np.arange(.5, 10, step):
                plt.pause(.00001)
                plt.clf()
                fig = system.render_trail(run, fig=fig, time=t)

    def testDoublePendulumTrailFast(self):
        self.clearFigs()
        run = self.pendulum_solver.run([0, 10], [.2, 1, 0, 3])
        system = DoublePendulumSystem()
        if False:
            fig = system.render_trail(run, time=.5)
            plt.show(False)
            step = .05
            for t in np.arange(.5, 10, step):
                plt.pause(.00001)
                plt.clf()
                fig = system.render_trail(run, fig=fig, time=t)

    def slowNBodyGradientCalc(self, v, m, body_dim):
        body_system = NBodySystem(m, body_dim)
        m = np.array(m)
        r, p = body_system.unpack_ham(v)

    def test3BodyGradientDataSet1(self):
        pass

    def test3BodyGradientDataSet2(self):
        pass

    def test3BodyGradientDataSet3(self):
        pass

    def test3BodyVaryMass(self):
        pass

    def test4BodyGradient(self):
        pass

    def test3Body2DGradient(self):
        pass

    def test4Body2DGradient(self):
        pass
