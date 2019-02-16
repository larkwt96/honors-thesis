from scipy.integrate import solve_ivp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class SystemSolver:
    def __init__(self, system):
        self.system = system
        self._runs = []

    def run(self, t_span, y0, method=None, max_step=np.inf):
        if method is None:
            method = self.system.method
        res = solve_ivp(self.system.fun, t_span, y0,
                        method=method, max_step=max_step)
        run = {
            'res': res,
            't_span': t_span,
            'y0': y0,
            'method': method,
            'max_step': max_step,
        }
        self._runs.append(run)
        return run

    def get_last_run(self):
        if len(self._runs) == 0:
            raise Exception("Expected res or last_run to be defined.")
        return self._runs[-1]

    def plot2d(self, res=None, *args, **kwargs):
        if res is None:
            res = self.get_last_run()
        res = res['res']

        fig = plt.figure()
        plt.plot(res.y[0, :], res.y[1, :], 'r-')
        plt.tight_layout()
        plt.show(*args, **kwargs)
        return plt, fig

    def plot3d(self, res=None, *args, **kwargs):
        if res is None:
            res = self.get_last_run()
        res = res['res']

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x, y, z = res.y
        ax.plot(x, y, z, '-b')
        plt.tight_layout()
        plt.show(*args, **kwargs)
        return plt, fig

    def plotnd(self, res=None, *args, **kwargs):
        if res is None:
            res = self.get_last_run()
        res = res['res']

        fig = plt.figure()
        num_plots = res.y.shape[0] + 1  # +1 for the overlay
        width = 9
        height = 3 * num_plots
        overlay = plt.subplot(num_plots, 1, num_plots)
        for i, y in enumerate(res.y):
            plt.subplot(num_plots, 1, i + 1)
            plt.plot(y)
            overlay.plot(y, label='{} dim'.format(i))
        plt.tight_layout()
        plt.show(*args, **kwargs)
        return plt, fig
