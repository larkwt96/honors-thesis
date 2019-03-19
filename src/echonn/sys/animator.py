import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.constants import pi
from abc import ABC, abstractmethod


class Animator(ABC):
    def __init__(self, trail_length=1, max_t=10):
        """
        trail_length in time, not dist
        max_t should be set by implementer
        """
        self.anim = None
        self.trail_length = trail_length
        self.ms_per_frame = 50
        self.max_t = max_t
        self.nFrames = self.calc_frames(max_t)

    @abstractmethod
    def init_plot(self):
        """
        This method is implemented and the line objects are returned in an array following the figure
        """
        fig = plt.figure()
        data, *_ = plt.plot([1, 2, 3])[0]
        # fig, self.lines = (figure, [line1, line2, line3, ...])
        return fig, [data]

    @abstractmethod
    def animator(self, framei):
        # t, y = self.get_data(framei)
        # do somethign with self.lines
        pass

    def calc_frames(self, max_t):
        return int(max_t * 1000 / self.ms_per_frame)

    def render(self):
        fig, self.lines = self.init_plot()
        self.anim = animation.FuncAnimation(fig,
                                            self.animator,
                                            frames=self.nFrames,
                                            interval=self.ms_per_frame,
                                            blit=False)

    def save(self, fname, ext='.gif'):
        """
        saves the file to the format specified by ext, default is recommended.
        if you'd like to override this setting, then set ext
        """
        self.anim.save(fname+ext)

    def render_ipynb(self):
        rc('animation', html='jshtml')
        rc('animation', embed_limit=50)
        return self.anim

    def get_data(self, frame_i, t, y):
        start_time, end_time = self.get_data_t_span(frame_i)
        mask = self.get_data_mask(start_time, end_time, t)
        return t[mask], y[mask]

    def get_data_t_span(self, frame_i):
        end_time = frame_i * self.ms_per_frame / 1000
        start_time = end_time - self.trail_length
        return start_time, end_time

    @staticmethod
    def get_data_mask(start_time, end_time, t):
        return np.where((start_time < t) & (t < end_time))
