import numpy as np
from .system import DynamicalSystem
from scipy.constants import g, pi
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class DoublePendulumSystem(DynamicalSystem):
    """
    Equal mass and equal length
    """

    def __init__(self, m=1, l=1):
        super().__init__(4, 'BDF')
        self.m = m
        self.l = l

        self._theta_pre = 6 / (m*l**2)
        self._gl = g / l

        self._moment_pre = -.5*m*l**2

    def fun(self, t, v):
        a1, a2, p1, p2 = v

        # a1p, a2p
        cos_a1a2 = np.cos(a1 - a2)
        theta_den = 16 - 9 * cos_a1a2**2
        factor = self._theta_pre / theta_den
        theta_num1 = 2 * p1 - 3 * cos_a1a2 * p2
        theta_num2 = 8 * p2 - 3 * cos_a1a2 * p1
        a1p = factor * theta_num1
        a2p = factor * theta_num2

        # p1p, p2p
        first_term = a1p * a2p * np.sin(a1-a2)
        p1p = self._moment_pre * (first_term + 3 * self._gl * np.sin(a1))
        p2p = self._moment_pre * (-first_term + self._gl*np.sin(a2))

        return a1p, a2p, p1p, p2p

    def get_endpoints(self, run):
        inner_theta, outer_theta, _, _ = run['results'].y
        inner_theta = inner_theta - pi/2
        outer_theta = outer_theta - pi/2
        inner_pos = np.array([np.cos(inner_theta), np.sin(inner_theta)])
        outer_relative_pos = np.array([np.cos(outer_theta),
                                       np.sin(outer_theta)])
        outer_pos = inner_pos + outer_relative_pos
        return inner_pos, outer_pos

    def render_fade_trail(self, run, fig=None, limit_margin=.1, time=1):
        # get figure
        if fig is None:
            fig = plt.figure()
        else:
            plt.figure(fig.number)

        # get data from run
        t = run['results'].t
        inner_pos, outer_pos = self.get_endpoints(run)
        limit_size = 2*run['system'].l+limit_margin
        limits = [-limit_size, limit_size, -limit_size, limit_size]
        mask = np.where((time - 1 <= t) & (t <= time))

        # apply mask
        t = t[mask]
        if t[-1] == 0 or t.shape[0] <= 2:
            return fig
        t_norm = (t - t[0]) / t[-1]
        inner_pos = inner_pos[:, mask].reshape(2, -1)
        outer_pos = outer_pos[:, mask].reshape(2, -1)

        # draw
        plt.axis(limits)
        inner_plots = []
        outer_plots = []
        alphas = []
        for i in range(t_norm.shape[0] - 1):
            inner_plots.append(plt.plot(*inner_pos[:, i:i+2])[0])
            outer_plots.append(plt.plot(*outer_pos[:, i:i+2])[0])
            alphas.append(t_norm[i])

        inner_c = inner_plots[0].get_color()
        outer_c = outer_plots[0].get_color()
        zipped_line_alphas = zip(alphas, inner_plots, outer_plots)
        for alpha, inner_plot, outer_plot in zipped_line_alphas:
            # set inner color
            inner_rgba = matplotlib.colors.to_rgba(inner_c, alpha)
            inner_plot.set_color(inner_rgba)

            # set outer color
            outer_rgba = matplotlib.colors.to_rgba(outer_c, alpha)
            outer_plot.set_color(outer_rgba)

        pendulum = np.array(
            [[0, 0], [*inner_pos[:, -1]], [*outer_pos[:, -1]]]).T
        plt.plot(*pendulum, color='black')
        plt.scatter(*pendulum, color='black', s=10)
        return fig

    def render_trail(self, run, fig=None, limit_margin=.1, time=1):
        # get figure
        if fig is None:
            fig = plt.figure()
        else:
            plt.figure(fig.number)

        # get data from run
        t = run['results'].t
        inner_pos, outer_pos = self.get_endpoints(run)
        limit_size = 2*run['system'].l+limit_margin
        limits = [-limit_size, limit_size, -limit_size, limit_size]
        mask = np.where((time - 1 <= t) & (t <= time))

        # apply mask
        t = t[mask]
        if t[-1] == 0 or t.shape[0] <= 2:
            return fig
        inner_pos = inner_pos[:, mask].reshape(2, -1)
        outer_pos = outer_pos[:, mask].reshape(2, -1)

        # draw
        plt.axis(limits)
        plt.plot(*inner_pos)
        plt.plot(*outer_pos)

        pendulum = np.array([[0, 0],
                             [*inner_pos[:, -1]],
                             [*outer_pos[:, -1]]]).T
        plt.plot(*pendulum, color='black')
        plt.scatter(*pendulum, color='black', s=10)
        return fig

    def render_path(self, run, fig=None, limit_margin=.1, dot_size=25):
        # get data from run
        t = run['results'].t
        _, _, inner_moment, outer_moment = run['results'].y
        inner_pos, outer_pos = self.get_endpoints(run)

        # draw
        limit_size = 2*run['system'].l+limit_margin
        limits = [-limit_size, limit_size, -limit_size, limit_size]
        if fig is None:
            fig = plt.figure()
            for g in gridspec.GridSpec(3, 1, height_ratios=[5, 1, 1]):
                plt.subplot(g)
        else:
            plt.figure(fig.number)
        axes = fig.get_axes()

        # render
        axes[0].scatter([0], [0], s=10, c='black')
        axes[0].plot(*inner_pos, 'b-', label='inner pos', zorder=10)
        axes[0].plot(*outer_pos, 'r-', label='outer pos', zorder=11)
        axes[0].scatter(*inner_pos, s=dot_size, c=t/t[-1], zorder=0)
        axes[0].scatter(*outer_pos, s=dot_size, c=t/t[-1], zorder=1)
        axes[0].axis(limits)
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].legend()

        axes[1].plot(t, inner_moment, 'b-', label='inner moment', zorder=1)
        axes[1].scatter(t, inner_moment, s=dot_size, c=t/t[-1], zorder=0)
        axes[1].legend()

        axes[2].plot(t, outer_moment, 'r-', label='outer moment', zorder=1)
        axes[2].scatter(t, outer_moment, s=dot_size, c=t/t[-1], zorder=0)
        axes[2].legend()

        plt.tight_layout()
        return fig
