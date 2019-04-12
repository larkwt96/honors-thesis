import unittest
from echonn.ml import ESNExperiment
from echonn.sys import LorenzSystem, DoublePendulumSystem, RestrictedCircular3Body


class TestExperiment(unittest.TestCase):
    def testLorenz(self):
        experiment = ESNExperiment(LorenzSystem, 1000)
        # experiment.run()

    def testDoubPend(self):
        experiment = ESNExperiment(DoublePendulumSystem, 1000)
        # experiment.run()

    def test3BodyProblem(self):
        experiment = ESNExperiment(RestrictedCircular3Body, 1000)
        # experiment.run()
