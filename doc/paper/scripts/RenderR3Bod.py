import numpy as np
import matplotlib.pyplot as plt
from echonn.sys import SystemSolver, RestrictedCircular3Body, LyapunovSystem

if __name__ == "__main__":
    mu = 0.5
    sys = RestrictedCircular3Body(body_ratio=mu)
    lce = LyapunovSystem(sys)

    #init = [1, -1, .1, .1]
    init = np.array(1000*np.random.rand(4), dtype=int)/1000
    slv = SystemSolver(sys)
    #run = slv.run([0, 10], init, max_step=0.001)
    T = 50
    lce, run = slv.get_lce(T=T, y0=init)
    #lce, run = slv.get_lce(T=T)
    mat = run['results'].y[4:, -1].reshape(4, 4)
    run['results'].y = run['results'].y[[0, 2], :]
    slv.plot2d(run)
    plt.scatter([-sys.alpha, sys.mu], [0, 0])
    print(init)
    print(mat)
    print('lce:', lce)
    #plt.xlim((-15, 15))
    #plt.ylim((-15, 15))
    plt.show(True)
