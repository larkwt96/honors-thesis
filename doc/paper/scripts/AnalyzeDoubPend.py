import os
import pickle
from echonn.sys import DoublePendulumSystem
from echonn.ml import ESNExperiment


fn = os.path.join('.', 'doub_pend_results.p')
print('Testing Double Pendulum Model')
experiment = ESNExperiment(DoublePendulumSystem())
res = experiment.run()
with open(fn, 'wb') as f:
    pickle.dump(res, f)
print(res)
