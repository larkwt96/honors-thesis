import numpy as np
from numpy import linalg as LA
from .tools import init_weights


class EchoStateNetwork:
    def __init__(self, K, N, L, T0=100, alpha=.999, use_noise=False, sparse=False, f=None, g=None, g_inv=None):
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
        self.sparse = sparse  # TODO make weight matrix sparse
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
                raise Exception('g_inv must be specified')
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

    def scale_matrix(self, W):
        if W.shape[1] == 0:
            return W  # no matrix to scale
        return W / np.sqrt(W.shape[1])

    def normalize_weight(self):
        eigenvalues, _ = LA.eig(self.W)
        max_eigenvalue = max(eigenvalues)
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

        system_state = np.concatenate(u[n], x[n])  # not y[n-1]
        g(Wout @ system_state)

    def predict(self, us=None, Tf=None):
        if us is None and Tf is None:
            raise Exception('Must specify Tf or us')
        # format us
        if us is None:
            us = np.zeros((Tf-self.T, self.K))
        else:
            us = np.array(us).reshape(-1, self.K)
        # format Tf
        if Tf is None:
            Tf = self.T + us.shape[0]
        else:
            zeros = np.zeros((Tf - self.T - us.shape[0], self.K))
            us = np.vstack((us, zeros))
        self.u = np.vstack((self.u, us))
        # predict
        for n in range(self.n+1, Tf+1):
            self.n = n
            np.append(self.x, self.calc_x(n), axis=0)
            np.append(self.y, self.calc_y(n), axis=0)

    def train(self, ds, us=None):
        """
        u must be indexable and store a numpy array. if K =0 or u = 0 for all
        n, then us can be omitted.
        """
        # fill missing points with zeros
        ds = [np.zeros(self.L) if d is None else np.array(d) for d in ds]
        # convert to numpy
        ds = np.array(ds).reshape(len(ds), self.L)

        # reshape or fill passed input data
        if us is None:
            us = np.zeros((ds.shape[0], self.K))
        else:
            us = [np.zeros(self.K) if u is None else np.array(u) for u in us]
            us = np.array(us).reshape(len(us), self.K)
        self.u = np.insert(us, 0, 0, axis=0)  # make 1 indexed

        self.y = np.zeros_like(self.u)  # build output layer with same shape
        # test the shape of passed ds
        # fill trailing missing points with zeros also
        self.y[1:1+ds.shape[0]] = ds
        self.x = [np.zeros(self.N)]

        # step 1: initialize weights
        # done in init

        # step 2
        # sample the network with forced teacher
        # M = state (time x state)
        # T = target (time x target)
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
        self.Wout[:, :] = (LA.inv(M) @ T).T


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
