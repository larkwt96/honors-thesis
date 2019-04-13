import itertools
import os
import unittest
from echonn.ml import ESNExperiment
from echonn.sys import LorenzSystem, DoublePendulumSystem, RestrictedCircular3Body
import pickle


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.file_io = True
        self.verbose = True
        self.alpha = [.7, .8]
        self.N = [3]
        self.T0 = [500]
        self.params = itertools.product(self.alpha, self.N, self.T0)
        self.trials = 2

    @unittest.skip
    def testLorenz(self):
        if self.verbose:
            print()
        fn = os.path.join('src', 'test', 'test_data', 'lorenz.p')
        if self.verbose:
            print('Testing Lorenz Model')
        experiment = ESNExperiment(
            LorenzSystem(), self.params, self.trials, 1000)
        res = experiment.run(self.verbose)
        if self.file_io:
            with open(fn, 'wb') as f:
                pickle.dump(res, f)
            with open(fn, 'rb') as f:
                res = pickle.load(f)
        print(res)

    @unittest.skip
    def testDoubPend(self):
        if self.verbose:
            print()
        fn = os.path.join('src', 'test', 'test_data', 'doub_pend.p')
        if self.verbose:
            print('Testing Double Pendulum Model')
        experiment = ESNExperiment(
            DoublePendulumSystem(), self.params, self.trials, 1000)
        res = experiment.run(self.verbose)
        if self.file_io:
            with open(fn, 'wb') as f:
                pickle.dump(res, f)
            with open(fn, 'rb') as f:
                res = pickle.load(f)
        print(res)

    @unittest.skip
    def test3BodyProblem(self):
        if self.verbose:
            print()
        fn = os.path.join('src', 'test', 'test_data', '3b.p')
        if self.verbose:
            print('Testing 3 Body Problem Model')
        experiment = ESNExperiment(
            RestrictedCircular3Body(), self.params, self.trials, 1000)
        res = experiment.run(self.verbose)
        if self.file_io:
            with open(fn, 'wb') as f:
                pickle.dump(res, f)
