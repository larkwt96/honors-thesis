import itertools
import os
import pickle
from echonn.sys import LorenzSystem
from echonn.ml import ESNExperiment


fn = os.path.join('.', 'lorenz_results.p')
print('Testing Lorenz Model')
alphas = [.7, .75, .8, .85, .9, .98]
#alphas = [.7, .75, .8]
Ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300]
#Ns = [100, 150, 200]
T0s = [50, 100, 150, 200, 300, 500]
#T0s = [300]
params = list(itertools.product(alphas, Ns, T0s))
experiment = ESNExperiment(LorenzSystem(), params=params,
                           t_len=50, trials=3, time_steps_per_lce_time=100)
res = experiment.run(verbose=True)
with open(fn, 'wb') as f:
    pickle.dump(res, f)
print(res)
