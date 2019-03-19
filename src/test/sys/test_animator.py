import unittest
import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt
from echonn.sys import Animator
import os


class ExampleAnimator(Animator):
    def __init__(self, max_step=10):
        t = np.arange(0, max_step, .01)
        y = np.sin(2*pi*t)
        super().__init__(t, y, 1)
        self.max_step = max_step

    def init_plot(self):
        fig = plt.figure()
        plt.xlim((0, self.max_step))
        plt.ylim((-2, 2))
        t, y = self.get_data(0)
        line, *_ = plt.plot(t, y)
        return fig, [line]

    def animator(self, framei):
        t, y = self.get_data(framei)
        data, = self.lines
        data.set_data(t, y)


class TestAnimator(unittest.TestCase):
    def testAnimation(self):
        animator = ExampleAnimator()
        animator.render()
        fname = os.path.join('src', 'test', 'test_data', 'test_video')
        animator.save(fname)
        self.assertTrue(os.path.isfile(fname+'.gif'))
