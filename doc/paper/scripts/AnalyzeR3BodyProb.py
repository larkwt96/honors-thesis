import os
import pickle
import itertools
from echonn.sys import RestrictedCircular3Body
from echonn.ml import ESNExperiment


fn = os.path.join('.', 'rc3_body_results.p')
fn_y0s = os.path.join('.', 'rc3_body_y0s.p')
print('Testing Restricted Circular 3 Body Model')
alphas = [.7, .75, .8, .85, .9, .98]
#alphas = [.7, .8]
Ns = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300]
#Ns = [3]
T0s = [50, 100, 150, 200, 300, 500]
#T0s = [30]
params = list(itertools.product(alphas, Ns, T0s))
if os.path.exists(fn_y0s):
    with open(fn_y0s, 'rb') as f:
        y0s = pickle.load(f)
else:
    y0s = []
sys = RestrictedCircular3Body(search=True, body_ratio=0.5, max_tries=200,
                              T=500, rmin=0.001, rmax=10)
y0 = None
y0 = y0s[-1]
experiment = ESNExperiment(sys, y0=y0, params=params, use_diff=False,
                           t_len=50, trials=3,
                           time_steps_per_lce_time=100)
res = experiment.run(verbose=True)
y0 = res[0]['results'].y[:, 0]
y0s.append(y0)
with open(fn, 'wb') as f:
    pickle.dump(res, f)
with open(fn_y0s, 'wb') as f:
    pickle.dump(y0s, f)

print(res)
print('y0s', y0s)
