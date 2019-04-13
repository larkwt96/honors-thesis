import os
import pickle
from echonn.sys import DoublePendulumSystem
from echonn.ml import ESNExperiment


fn = os.path.join('.', 'lorenz_results.p')
print('Testing Lorenz Model')
experiment = ESNExperiment(DoublePendulumSystem())
res = experiment.run()
with open(fn, 'wb') as f:
    pickle.dump(res, f)
print(res)
