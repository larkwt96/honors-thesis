import itertools
import os
import pickle
from echonn.sys import DoublePendulumSystem
from echonn.ml import ESNExperiment


fn = os.path.join('.', 'doub_pend_results.p')
print('Testing Double Pendulum Model')
alphas = [.7, .75, .8, .85, .9, .98]
Ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300]
T0s = [50, 100, 150, 200, 300, 500]
params = list(itertools.product(alphas, Ns, T0s))
experiment = ESNExperiment(DoublePendulumSystem(), params=params, use_diff=False,
                           t_len=50, trials=3, time_steps_per_lce_time=100)
res = experiment.run(verbose=True)
with open(fn, 'wb') as f:
    pickle.dump(res, f)
print(res)
