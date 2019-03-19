import unittest
import os
import sys
from contextlib import contextmanager

import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt
from echonn.sys import Animator


class ExampleAnimator(Animator):
    def __init__(self, max_step=10):
        super().__init__()
        self.t = np.arange(0, max_step, .1)
        self.y = np.sin(2*pi*self.t)
        self.max_step = max_step

    def init_plot(self):
        fig = plt.figure()
        plt.xlim((0, self.max_step))
        plt.ylim((-2, 2))
        t, y = self.get_data(0, self.t, self.y)
        line, *_ = plt.plot(t, y)
        return fig, [line]

    def animator(self, framei):
        t, y = self.get_data(framei, self.t, self.y)
        data, = self.lines
        data.set_data(t, y)


class TestAnimator(unittest.TestCase):
    def testAnimation(self):
        animator = ExampleAnimator()
        animator.render()
        fname = os.path.join('src', 'test', 'test_data', 'test_video')

        # http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/

        @contextmanager
        def suppress_stdout():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

        try:
            os.remove(fname+'.gif')
        except:
            pass  # we don't care

        self.assertFalse(os.path.isfile(fname+'.gif'))
        with suppress_stdout():
            animator.save(fname)
        self.assertTrue(os.path.isfile(fname+'.gif'))
