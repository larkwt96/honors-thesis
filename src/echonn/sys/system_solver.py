import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from .lyapunov_system import LyapunovSystem


class SystemSolver:
    def __init__(self, system):
        self.system = system
        self._runs = []

    def run(self, t_span, y0, *args, **kwargs):
        """
        returns a dictionary with results.
        {
            'results': the results from solve_ivp
            'index': the index of the run in the solver's past runs
            'system': the system object used
            't_span': the time interval used in solve_ivp
            'y0': initial condition
            'args': args passed to solve_ivp
            'kwargs': kwargs passed to solve_ivp
        }
        """
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

    def build_lce_y0(self, y0, T, kwargs):
        if y0 is None:
            y0_use = np.random.rand(self.system.dim)
            run = self.run([0, T/2], y0_use, **kwargs)
            # tend it towards the attractor
            y0_use = run['results'].y[:, -1]
        else:
            y0_use = y0
        return y0_use

    def calc_lce(self, Df_y0, T):
        # calculate singular value decomposition
        _, svds, *_ = np.linalg.svd(Df_y0)
        svd = svds[0]
        if svd <= 0:
            return -np.inf
        # calculate lyapunov exponent
        lce = np.log(svd)/T
        return lce

    def quick_lce(self, T=None, y0=None, partition=None, **kwargs):
        """
        y0 - will be set randomly and then run through the system for a
        little
        T - should be set manually to a reasonable value.
        partition - compute to T in partitions to save memory (for faster
        computation) (defaults to T)
        kwargs - are passed to the run parameters of SystemSolver
        """
        if T is None:
            T = self.system.best_lce_T
        if partition is None:
            partition = T
        sys_dim = self.system.dim
        # create the lyapunov system
        lce_system = LyapunovSystem(self.system)
        # create a solver for the system
        lce_solver = SystemSolver(lce_system)
        # create an initial condition (or use given)
        y0_use = self.build_lce_y0(y0, T, kwargs)
        # create initial conditino for lyapunov solver
        v0 = lce_system.build_y0(y0_use)

        tspan = np.arange(0, T, T/partition)
        v = v0
        lces = []
        for i in range(1, tspan.shape[0]):
            # run lyapunov system
            run = lce_solver.run(tspan[i-1:i+1], v, **kwargs)
            # if failed, raise exception
            if run['results'].status != 0:
                raise Exception(run['results'].message)
            v = run['results'].y[:, -1]
            Df_y0 = v[sys_dim:].reshape(sys_dim, sys_dim)
            lce = self.calc_lce(Df_y0, tspan[i])
            lces.append(lce)
            print(tspan[i], lce)
        # run lyapunov system
        run = lce_solver.run([tspan[-1], T], v, **kwargs)
        # if failed, raise exception
        if run['results'].status != 0:
            print('Warning:', run['results'].message)
        vf = run['results'].y[:, -1]
        Df_y0 = v[sys_dim:].reshape(sys_dim, sys_dim)
        lce = self.calc_lce(Df_y0, T)
        print(float(T), lce)
        lces.append(lce)
        tspan[:-1] = tspan[1:]
        tspan[-1] = T
        return tspan, lces

    def get_lce(self, T=None, y0=None, **kwargs):
        """
        T - should be set manually to a reasonable value.
        y0 - will be set randomly and then run through the system for a
        little
        kwargs - are passed to the run parameters of SystemSolver
        """
        if T is None:
            T = self.system.best_lce_T
        sys_dim = self.system.dim
        # create the lyapunov system
        lce_system = LyapunovSystem(self.system)
        # create a solver for the system
        lce_solver = SystemSolver(lce_system)
        # create an initial condition (or use given)
        y0_use = self.build_lce_y0(y0, T, kwargs)
        # create initial conditino for lyapunov solver
        v0 = lce_system.build_y0(y0_use)
        # run lyapunov system
        run = lce_solver.run([0, T], v0, **kwargs)
        # if failed, raise exception
        if run['results'].status != 0:
            raise Exception(run['results'].message)
        # calculate the jacobian of trajectory with respect to initial condition
        y = run['results'].y
        Df_y0 = y[sys_dim:, -1].reshape(sys_dim, sys_dim)
        lce = self.calc_lce(Df_y0, T)
        return lce, run

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

    def plotnd(self, run=None, fig=None, figsize=None, legend=False, dims=None, overlay=True, title=None):
        """ warning legend=True is going to mess up the layout. The legend can be seen by the side of the mini graph. """
        run = self.get_last_run(run)
        sys_num = run['index']
        res = run['results']

        if fig is None:
            max_rows = 8
            num_plots = res.y.shape[0]
            if overlay:
                num_plots += 1
            rows = min(max_rows, num_plots)
            cols = max(1, math.ceil(num_plots/max_rows))
            if num_plots > max_rows:
                figsize = (7*cols/2, 5*rows/4)
            fig = plt.figure(figsize=figsize)
            for i in range(num_plots):
                plt.subplot(rows, cols, i + 1)
        else:
            fig = plt.figure(fig.number)
        plots = fig.get_axes()
        if overlay:
            overlay_plt = plots[-1]
            overlay_plt.set_ylabel('overlay')
        for i, y in enumerate(res.y):
            for _ in range(i+1):
                plots[i].plot(res.t, y)  # TODO: fix this hack
                # This makes graphs match color in overlay
            if dims is None:
                plots[i].set_ylabel('dim {}'.format(i))
            else:
                plots[i].set_ylabel(dims[i])
            label = 'sys {} dim {}'.format(sys_num, i)
            if overlay:
                overlay_plt.plot(res.t, y, label=label)
        if legend and overlay:
            overlay.legend()
        if title is not None:
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            plt.tight_layout()
        return fig
