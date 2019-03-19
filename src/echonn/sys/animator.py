import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.constants import pi
from abc import ABC, abstractmethod


class Animator(ABC):
    def __init__(self, t, y, trail_length):
        """
        trail_length in time, not dist
        """
        self.t = t
        self.y = y
        self.anim = None
        self.trail_length = trail_length
        self.ms_per_frame = 50

    @abstractmethod
    def init_plot(self):
        """
        This method is implemented and the line objects are returned in an array following the figure
        """
        fig = plt.figure()
        data, *_ = plt.plot([1, 2, 3])[0]
        return fig, [data]

    @abstractmethod
    def animator(self, framei):
        t, y = self.get_data(framei)
        # do somethign with self.lines

    def render(self):
        # plotter stuff
        nFrames = int(self.t[-1] * 1000 / self.ms_per_frame)
        fig, self.lines = self.init_plot()
        self.anim = animation.FuncAnimation(fig,
                                            self.animator,
                                            frames=nFrames,
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

    def get_data(self, frame_i):
        end_time = frame_i * self.ms_per_frame / 1000
        start_time = end_time - self.trail_length
        mask = np.where((start_time < self.t) & (self.t < end_time))
        return self.t[mask], self.y[mask]
