from scipy.integrate import solve_ivp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class SystemSolver:
    def __init__(self, system):
        self.system = system
        self._runs = []

    def run(self, t_span, y0, *args, **kwargs):
        method_given = len(args) > 0 or 'method' in kwargs
        system_has_method = self.system.method is not None
        if not method_given and system_has_method:
            kwargs['method'] = self.system.method
        results = solve_ivp(self.system.fun, t_span, y0, *args, **kwargs)
        run = {
            'results': results,
            'index': len(self._runs),
            'system': self.system,
            't_span': t_span,
            'y0': y0,
            'args': args,
            'kwargs': kwargs,
        }
        self._runs.append(run)
        return run

    def get_last_run(self, run=None):
        if run is None:
            if len(self._runs) == 0:
                raise Exception("Expected run or a last run to be defined.")
            else:
                return self._runs[-1]
        else:
            return run

    def plot2d(self, run=None, fig=None):
        run = self.get_last_run(run)
        res = run['results']
        if fig is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig.number)
        plt.plot(*res.y, '-')
        plt.tight_layout()
        return fig

    def plot3d(self, run=None, fig=None):
        run = self.get_last_run(run)
        res = run['results']
        if fig is None:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
        else:
            fig = plt.figure(fig.number)
            ax,  = fig.get_axes()
        ax.plot(*res.y, '-')
        plt.tight_layout()
        return fig

    def plotnd(self, run=None, fig=None):
        run = self.get_last_run(run)
        sys_num = run['index']
        res = run['results']

        if fig is None:
            fig = plt.figure()
            num_plots = res.y.shape[0] + 1  # +1 for the overlay
            for i in range(num_plots):
                plt.subplot(num_plots, 1, i + 1)
        else:
            fig = plt.figure(fig.number)
        plots = fig.get_axes()
        overlay = plots[-1]
        overlay.set_ylabel('overlay')
        for i, y in enumerate(res.y):
            for _ in range(i+1):
                plots[i].plot(res.t, y)  # TODO: fix this hack
                # This makes graphs match color in overlay
            plots[i].set_ylabel('dim {}'.format(i))
            label = 'sys {} dim {}'.format(sys_num, i)
            overlay.plot(res.t, y, label=label)
        overlay.legend()
        plt.tight_layout()
        return fig
