from echonn.ml import EchoStateNetwork
import unittest


class TestEchoStateNetwork(unittest.TestCase):
    def testFail(self):
        pass  # self.fail('haha')
    # def __init__(self, K, N, L, T0=100, alpha=.999, use_noise=False, sparse=False, f=None, g=None)
    # test sparse works

    # def init_weights(self):
    # test echo state property

    # def scale_matrix(self, W):
    # test scales properly
    # test with zero length matrix

    # def normalize_weight(self):
    # test with manually set matrix

    # def calc_x(self, n):
    # def calc_y(self, n):
    # def predict(self, us=None, Tf=None):
    # def train(self, us, ds):
    # test following with sin
    # test K = 0
    # test K = 1
    # test K = 10
    # test L = 1
    # test L = 10
    # test N = 1
    # test N = 10
    # test T0 = 1
    # test T0 = 10
    # test T0 = 100
    # test noise works
