import os
import sys
import unittest
from contextlib import contextmanager
from echonn.sys.double_pendulum_animator import DoublePendulumAnimator
from echonn.sys import DoublePendulumSystem, SystemSolver


class TestDoubPenAnim(unittest.TestCase):
    def setUp(self):
        # test data
        self.pendulum_system = DoublePendulumSystem()
        self.pendulum_solver = SystemSolver(self.pendulum_system)
        # run render test
        self.render_til = 10
        self.render_long = False  # slows test a lot
        self.render = False  # slows test

    def testLongRun(self):
        tf = 50
        run1 = self.pendulum_solver.run([0, tf], [.2, 1, 1, 0])
        run2 = self.pendulum_solver.run([0, tf], [.2, 1.1, 1, 0])
        if self.render_long:
            self.runRender([run1, run2], 'long_run', mult=tf/10)

    def testRender(self):
        run1 = self.pendulum_solver.run([0, self.render_til], [.2, 1, 1, 0])
        run2 = self.pendulum_solver.run([0, self.render_til], [.2, 1.1, 1, 0])
        if self.render:
            self.runRender(run1, 'run1')
            self.runRender([run1], 'run1_mult', mult=2)
            self.runRender([run1, run2], 'run2', mult=.5)

    def runRender(self, runs, fname, mult=1):
        animator = DoublePendulumAnimator(runs, speed=mult)
        animator.render()
        fname = os.path.join('src', 'test', 'test_data', fname)

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
