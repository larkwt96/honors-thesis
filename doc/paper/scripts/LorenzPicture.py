import os
import numpy as np
import matplotlib.pyplot as plt
from echonn.sys import LorenzSystem, SystemSolver

if __name__ == "__main__":
    lorenz_3d_plot = os.path.join('..', 'images', 'lorenz_3d_plot.png')
    lorenz_nd_plot = os.path.join('..', 'images', 'lorenz_nd_plot.png')
    slv = SystemSolver(LorenzSystem())
    res = slv.run([0, 50], [10, 20, 30])
    slv.plotnd(res, dims=['x', 'y', 'z'])
    plt.savefig(lorenz_nd_plot)
    slv.plot3d(res)
    plt.savefig(lorenz_3d_plot)
    plt.show(True)
