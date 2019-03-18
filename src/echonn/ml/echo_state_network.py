import numpy as np
from numpy import linalg as LA
from .tools import init_weights
from .time_series_forecaster import TimeSeriesForecaster


class EchoStateNetwork(TimeSeriesForecaster):
    def __init__(self, K, N, L, T0=50, alpha=.8, use_noise=False, f=None, g=None, g_inv=None):
        """
        K - input units, u
        N - internal units, x
        L - output units, y
        T0 - how long does it take for transient dynamics to wash out (10 for
        small, fast nets to 500 for large, slow nets). T is the sampling range, set by the train call.
        use_noise - should the algorithm feed noise into the model during sampling (0-T).
        """
        self.shapes = [
            (N, K),
            (N, N),
            (L, K+N),
            (N, L)
        ]
        self.K = K
        self.N = N
        self.L = L
        self.T0 = T0  # at T0, x[T0] has stabilized
        self.alpha = alpha
        self.use_noise = use_noise  # TODO add noise to sampling
        self.w = None
        self.Ws = None
        self.Win = None
        self.W = None
        self.Wout = None
        self.Wback = None
        self.f = np.tanh if f is None else f
        if g is None:
            self.g = lambda x: x
            self.g_inv = self.g
        else:
            self.g = g
            if g_inv is None:
                raise Exception('g_inv must be specified with g')
            self.g_inv = g_inv

        # set on train
        self.n = 0
        self.T = 0
        self.init_weights()  # max eigenvalue set to alpha
        self.u = [np.zeros(self.K)]
        self.x = [np.zeros(self.N)]
        self.y = [np.zeros(self.L)]

    def init_weights(self):
        self.w, self.Ws = init_weights(self.shapes)
        self.Win, self.W, self.Wout, self.Wback = self.Ws
        self.normalize_weight()  # max eigenvalue set to alpha

    def normalize_weight(self):
        eigenvalues, _ = LA.eig(self.W)
        max_eigenvalue = max(np.absolute(eigenvalues))
        self.Win[:, :] = self.scale_matrix(self.Win)
        self.Wout[:, :] = self.scale_matrix(self.Wout)
        self.Wback[:, :] = self.scale_matrix(self.Wback)
        self.W *= self.alpha / max_eigenvalue

    def calc_x(self, n):
        """
        x is 1 indexed (actually, [0] = 0)
        x(n) = f(Win u(n) + W x(n-1) + Wback y(n-1))
        """
        f = self.f
        Win = self.Win
        W = self.W
        Wback = self.Wback
        u = self.u
        x = self.x
        y = self.y

        return f(Win @ u[n] + W @ x[n-1] + Wback @ y[n-1])

    def calc_y(self, n):
        """
        y is 1 indexed
        y(n) = gWout (u(n), x(n))) (tuple = concat)
        """
        g = self.g
        Wout = self.Wout
        u = self.u
        x = self.x

        system_state = np.concatenate((u[n], x[n]))
        return g(Wout @ system_state)

    def score(self, ds, ys, T, Tf=None, T0=None):
        """
        ys should be what was returned by predict
        """
        if T0 is None:
            T0 = self.T0
        ds = self.fill_with_zeros(ds, self.L)
        ys = self.fill_with_zeros(ys, self.L)
        train_d = ds[T0:T]
        train_y = ys[T0:T]
        train_rmse = self.rmse(train_d, train_y)
        test_d = ds[T:Tf]
        test_y = ys[T:Tf]
        test_rmse = self.rmse(test_d, test_y)
        return (train_d, train_y, train_rmse), (test_d, test_y, test_rmse)

    def predict(self, ds=None, us=None, Tf=None):
        """
        ds defaults to an empty array and there will be no data to initialize
        the reservoir. It will use the provided ds until it runs out. Then,
        it will calculate ds from the model until it reaches Tf.

        us defaults to the zero array extended to reach Tf
        (this is for when the model doesn't use input data.)

        Tf defaults to the length of us but can be set manually. It will stop
        at Tf. If Tf is larger than u, then u is filled with zeros. If Tf is
        less than u, it will stop before the end of u is reached (so you can
        specify more us than Tfs).
        """
        # set y
        if ds is None:
            ds = np.zeros((0, self.L))
        else:
            ds = self.fill_with_zeros(ds, self.L)
        self.y = np.insert(ds, 0, 0, axis=0)

        # init u
        if us is None and Tf is None:
            raise Exception('Must specify Tf or us')

        if us is None:
            us = np.zeros((Tf, self.K))
        else:
            us = self.fill_with_zeros(us, self.K)

        if Tf is None:
            Tf = us.shape[0]
        else:
            num_us = max(Tf, us.shape[0])
            us_new = np.zeros((num_us, self.K))
            us_new[:us.shape[0]] = us
            us = us_new
        self.u = np.insert(us, 0, 0, axis=0)

        # init x
        self.x = np.zeros((1, self.N))

        # feed x and y
        # predict
        for n in range(1, Tf+1):
            self.n = n
            self.x = np.append(self.x, [self.calc_x(n)], axis=0)
            if self.y.shape[0] <= n:
                self.y = np.append(self.y, [self.calc_y(n)], axis=0)
        return self.y[1:]

    @staticmethod
    def fill_with_zeros(vs, dim):
        vs = [np.zeros(dim) if v is None else np.array(v) for v in vs]
        return np.array(vs).reshape(len(vs), dim)

    def fit(self, ds, us=None, reinit_weights=False):
        """
        us defaults to zeros the length of ds. This is useful when the model
        doesn't use input data. if use is provided then it must be the same
        length as ds.

        reinit_weights set to True will reinitialize the weights to random
        before fitting. Otherwise, multiple runs does nothing (but reset the
        output layer to the same calculated solution).
        """
        # step 1: initialize weights
        if reinit_weights:
            self.init_weights()

        # step 2
        ds = self.fill_with_zeros(ds, self.L)
        if us is None:
            us = np.zeros((ds.shape[0], self.K))
        else:
            us = self.fill_with_zeros(us, self.K)
        if ds.shape[0] != us.shape[0]:
            raise Exception('ds and us shapes don\'t match')
        self.u = np.insert(us, 0, 0, axis=0)
        self.y = np.insert(ds, 0, 0, axis=0)

        # initialize network state
        self.x = [np.zeros(self.N)]
        for n in range(1, self.u.shape[0]):
            self.n = n
            self.x.append(self.calc_x(n))
        self.x = np.array(self.x).reshape(-1, self.N)

        self.T = self.n

        # T = 0 is at index 1, so T0 = 10 means T = 10 is good, so we want
        # index 11 and on.
        #
        # if len(u) is 100 and T0 = 10, then shape is 91. u[10+1:] ignores
        # first 10 points and the 0 row is added - 10 + 1 is -9. 100 - 9 is 91,
        # what we expect.

        # note that M stores u and x, but not d, since it's not hooked up
        # shape = (T - T0 + 1, N)
        M = np.hstack((self.u[self.T0+1:, :], self.x[self.T0+1:, :]))
        T = self.g_inv(self.y[self.T0+1:, :])  # shape = (T - T0 + 1, L)

        # step 3
        # solve for Wout = (inv(M) T).T
        self.Wout[:, :] = (LA.pinv(M) @ T).T


'''
The forecaster will take a series of x values and be able to reproduce them.
For now, I will assume equal steps in time. I can vary it and pass that value
as input, but that complicates the model

From Herbert Jaeger's paper (see references)

Network design:
Input layer (K units), u
Internal layer (N units), x
Output layer (L units), y

Actual Data (L units), d

The weights have the following shape:
NxK / NxN / Lx(K+N) / NxL
Win / W   / Wout    / Wback

Activation of x:
x(n+1) = f(Win u(n+1) + W x(n) + Wback y(n))

Activation of y:
y(n+1) = g(Wout (u(n+1), x(n+1)))
Note:
    * g is activation (usually, sigmoid)
    * We define f = tanh and g = tanh
    * sometimes you can pass y(n) too, but that's rare


I need a data splitter and an evaluator. I need to do a parameter search with
cross validation set.


It's possible to have K be 0. Then Win is Nx0 and Win @ u = np.zeros(N).

I want to be able to train it and use it.

To train it, I will have u and d.

To use it, I will have u and y0. However, it needs to "washout transient
dynamics", so y0-T0 will be innaccurate but after that y will converge to d.
'''
