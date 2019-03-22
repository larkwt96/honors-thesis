from echonn.ml import EchoStateNetwork
import numpy as np
import unittest


class TestEchoStateNetwork(unittest.TestCase):
    # def __init__(self, K, N, L, T0=100, alpha=.999, use_noise=False, sparse=False, f=None, g=None)
    def testInit(self):
        K = 10
        N = 11
        L = 12
        esn = EchoStateNetwork(K, N, L)
        self.assertEqual(esn.Win.shape, (N, K))
        self.assertEqual(esn.W.shape, (N, N))
        self.assertEqual(esn.Wback.shape, (N, L))
        self.assertEqual(esn.Wout.shape, (L, K+N))

    # def init_weights(self):
    # test echo state property

    def testWEigenAlpha(self):
        for alpha in [.7, .8, .85, .9, .95]:
            esn = EchoStateNetwork(10, 11, 12, alpha=alpha)
            eigs, _ = np.linalg.eig(esn.W / alpha)
            eig_val = max(np.absolute(eigs))
            self.assertAlmostEqual(1, eig_val)

    def testInitStateFadesWithTime(self):
        for alpha in [.7, .8, .85, .9, .95]:
            esn = EchoStateNetwork(1, 5, 1, alpha=alpha)
            ds = np.arange(500)
            ds[30:] = 0
            us = np.copy(ds)

            esn.predict(ds, us)
            self.assertNotAlmostEqual(0, np.sum(esn.x[30]))
            self.assertAlmostEqual(0, np.sum(esn.x[-1]))

    # def scale_matrix(self, W):
    # test scales properly
    # test with zero length matrix

    def testScaleMatrix(self):
        esn = EchoStateNetwork(10, 10, 10)
        W1x100 = np.ones((1, 100))
        W10x100 = np.ones((10, 100))
        W1x25 = np.ones((1, 25))
        W10x25 = np.ones((10, 25))
        for W in [W1x100, W10x100, W1x25, W10x25]:
            scale = 1 / np.sqrt(W.shape[1])
            scaled_W = esn.scale_matrix(W)
            self.assertAlmostEqual(np.mean(scaled_W), scale)

    def testScaleZeroMatrix(self):
        esn = EchoStateNetwork(10, 10, 10)
        W0x25 = np.ones((0, 25))
        W0x0 = np.ones((0, 0))
        W25x0 = np.ones((25, 0))
        for W0 in [W0x25, W0x0, W25x0]:
            # assert doesn't throw error
            self.assertEqual(W0.shape, esn.scale_matrix(W0).shape)

    # def normalize_weight(self):
    # test with manually set matrix
    def testNormalizeWeights(self):
        for alpha in [.7, .8, .85, .9, .95]:
            esn = EchoStateNetwork(10, 11, 12, alpha=alpha)
            eigs, _ = np.linalg.eig(esn.W)
            eig_val = max(np.absolute(eigs))
            self.assertGreater(1, eig_val)
            self.assertNotAlmostEqual(1, eig_val)

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

    def testKNLCombo(self):
        Ks = [0, 1, 10]
        Ns = [1, 10]
        Ls = [1, 10]
        T0s = [1, 10, 100]
        for K in Ks:
            for N in Ns:
                for L in Ls:
                    for T0 in T0s:
                        self.runKNLCombo(K, N, L, T0)

    def runKNLCombo(self, K, N, L, T0):
        Tf = 500
        T = 300
        us = np.ones((Tf, K))
        ds = np.ones((T, L))
        esn = EchoStateNetwork(K, N, L, T0=T0)
        means = []
        stds = []
        for _ in range(5):
            esn.fit(ds, us[:T], reinit_weights=True)
            esn.predict(ds[:T], us)
            means.append(np.mean(esn.y[Tf]))
            stds.append(np.std(esn.y[Tf]))
        # checking for compatibility not good convergence
        self.assertLess(abs(1 - np.mean(means)), 10)
        self.assertAlmostEqual(0, np.mean(stds), places=1)
        return esn

    def testUntrainedFails(self):
        Tf = 500
        T = 300
        K = 1
        N = 50
        L = 1
        T0 = 100
        us = np.ones((Tf, K))
        ds = np.ones((T, L))
        esn = EchoStateNetwork(K, N, L, T0=T0)
        esn.predict(ds, us)
        self.assertNotAlmostEqual(1, np.mean(esn.y[-1]), places=1)

    def testSuperFit(self):
        # might not always work, but if passes somewhat consistently proves something :)
        esn = self.runKNLCombo(5, 50, 3, 100)
        self.assertAlmostEqual(1, np.mean(esn.y[-1]))
        self.assertAlmostEqual(0, np.std(esn.y[-1]), places=2)

    def mse(self, x, y):
        x = x.reshape(-1)
        y = y.reshape(-1)
        return np.mean((x - y)**2)

    def runPaperExperiment(self, noise=0):
        esn = EchoStateNetwork(0, 20, 1, T0=100, g=np.tanh,
                               g_inv=np.arctanh, alpha=.99, noise=noise)
        n = np.arange(1, 351)
        ds = np.sin(n/4)/2
        esn.fit(ds[:300])

        # train measure
        ys = esn.predict(ds[:100], Tf=350)
        train_res, test_res = esn.score(ds, ys, 300, 350, 100)
        mse_train = self.mse(ds[100:300], ys[100:300])
        mse_test = self.mse(ds[300:], ys[300:])
        self.assertAlmostEqual(train_res[2]**2, mse_train)
        self.assertAlmostEqual(test_res[2]**2, mse_test)
        return mse_train, mse_test

    def testPaperExperiment(self):
        for _ in range(10):
            mse_train, mse_test = self.runPaperExperiment()
            if np.isclose(mse_train, 0, atol=10**-10) and np.isclose(mse_test, 0, atol=10**-10):
                return
        raise Exception('paper experiment failed')

    def testFitReInitWeights(self):
        esn = EchoStateNetwork(0, 10, 1, T0=100)
        ds = np.sin(np.arange(0, 50, .1)).reshape(-1, 1)
        W0 = np.copy(esn.W)
        esn.fit(ds, reinit_weights=True)
        W1 = np.copy(esn.W)
        esn.fit(ds, reinit_weights=True)
        W2 = np.copy(esn.W)
        esn.fit(ds, reinit_weights=False)
        W3 = np.copy(esn.W)

        mse01 = self.mse(W0, W1)
        mse12 = self.mse(W1, W2)
        mse23 = self.mse(W2, W3)

        self.assertNotAlmostEqual(0, mse01)
        self.assertNotAlmostEqual(0, mse12)
        self.assertAlmostEqual(0, mse23)

    # test noise works
    def testNoiseWorks(self):
        for _ in range(10):
            mse_train, mse_test = self.runPaperExperiment(noise=None)
            #print(mse_train, mse_test)
            if np.isclose(mse_train, 0, atol=10**-8) and np.isclose(mse_test, 0, atol=10**-8):
                return
        raise Exception('noise paper experiment failed')

    def testBiasWorks(self):
        pass  # TODO enable bias in esn
