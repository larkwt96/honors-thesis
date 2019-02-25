import unittest
import numpy as np
from echonn.sys import NBodySystem, SystemSolver


class TestModel(unittest.TestCase):
    def setUp(self):
        # test data
        self.n_body_system = NBodySystem()
        self.n_body_solver = SystemSolver(self.n_body_system)
        self.n_body_data = []
        self.n_body_expected = []

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
