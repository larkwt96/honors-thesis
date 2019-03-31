from .animator import Animator
import numpy as np
from scipy.constants import pi
import matplotlib
import matplotlib.pyplot as plt


class DoublePendulumAnimator(Animator):
    def __init__(self, runs, limit_margin=.1, speed=1):
        try:
            max_t = 0
            for run in runs:
                max_t = max(max_t, run['results'].t[-1] / speed)
        except TypeError:
            max_t = runs['results'].t[-1] / speed
            runs = [runs]
        super().__init__(1, max_t=max_t)
        self.runs = runs
        limit_size = -1
        for run in runs:
            l = run['system'].l1 + run['system'].l2
            limit_size = max(limit_size, l+limit_margin)
        self.limits = [-limit_size, limit_size, -limit_size, limit_size]
        self.speed = speed

    def init_plot(self):
        """
        This method is implemented and the line objects are returned in an array following the figure
        """
        self.fig = plt.figure()
        return self.fig, []

    def animator(self, frame_i):
        plt.figure(self.fig.number)
        plt.clf()
        for run in self.runs:
            t_span = self.get_data_t_span(frame_i)
            self.render_fade_trail(run, t_span)

    def get_endpoints(self, run):
        inner_theta, outer_theta, _, _ = run['results'].y
        l1 = run['system'].l1
        l2 = run['system'].l2
        inner_theta = inner_theta - pi/2
        outer_theta = outer_theta - pi/2
        inner_pos = l1*np.array([np.cos(inner_theta), np.sin(inner_theta)])
        outer_relative_pos = l2*np.array([np.cos(outer_theta),
                                          np.sin(outer_theta)])
        outer_pos = inner_pos + outer_relative_pos
        return inner_pos, outer_pos

    def render_fade_trail(self, run, t_span):

        # get data from run
        t = run['results'].t / self.speed
        mask = self.get_data_mask(*t_span, t)
        if np.sum(mask) == 0:
            mask = (np.array([0]))  # force at least initial position
        inner_pos, outer_pos = self.get_endpoints(run)

        # apply mask
        t = t[mask]
        if t.shape[0] == 0:
            t_norm = np.ones_like(t)
        else:
            t_shift = t - t[0]
            t_norm = np.divide(
                t_shift, t_shift[-1], out=np.ones_like(t), where=(t != 0))
        inner_pos = inner_pos[:, mask].reshape(2, -1)
        outer_pos = outer_pos[:, mask].reshape(2, -1)

        # draw
        fig = self.fig
        plt.figure(fig.number)
        plt.axis(self.limits)
        if t.shape[0] > 1:
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
