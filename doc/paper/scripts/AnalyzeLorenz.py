import os
import pickle
from echonn.sys import LorenzSystem
from echonn.ml import ESNExperiment


fn = os.path.join('.', 'lorenz_results.p')
print('Testing Lorenz Model')
experiment = ESNExperiment(LorenzSystem())
res = experiment.run(verbose=True)
with open(fn, 'wb') as f:
    pickle.dump(res, f)
print(res)
