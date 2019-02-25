import unittest

from echonn.sys import CircleSystem, LorenzSystem, SystemSolver


class TestModel(unittest.TestCase):
    def setUp(self):
        # test data
        self.circle_solver = SystemSolver(CircleSystem())
        self.lorenz_solver = SystemSolver(LorenzSystem())

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
