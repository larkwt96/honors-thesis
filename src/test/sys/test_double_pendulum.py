import unittest

import matplotlib.pyplot as plt
import numpy as np
from echonn.sys import DoublePendulumSystem, SystemSolver

from .sys_util import clearFigs


class TestModel(unittest.TestCase):
    def setUp(self):
        # test data
        self.pendulum_system = DoublePendulumSystem()
        self.pendulum_solver = SystemSolver(self.pendulum_system)

    def testDoublePendulum(self):
        clearFigs()
        run = self.pendulum_solver.run([0, 2], [.2, 1, 0, 0])
        run2 = self.pendulum_solver.run([0, 2], [.2, 1.1, 0, 0])
        fig = self.pendulum_solver.plotnd(run)
        self.pendulum_solver.plotnd(run2, fig)
        # plt.show(True)

        clearFigs()
        system = DoublePendulumSystem()
        fig = system.render_path(run, dot_size=2)
        system.render_path(run2, fig=fig, dot_size=2)
        # plt.show(True)

    def testDoublePendulumFade(self):
        clearFigs()
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
        clearFigs()
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
        clearFigs()
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
