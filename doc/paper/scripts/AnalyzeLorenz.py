import os
import pickle
from echonn.sys import LorenzSystem
from echonn.ml import ESNExperiment


fn = os.path.join('.', 'lorenz_results.p')
print('Testing Lorenz Model')
params = [
    (.7, 500, 300),
    (.8, 500, 300),
    (.9, 500, 300),
    (.98, 500, 300),
    (.7, 300, 300),
    (.8, 300, 300),
    (.9, 300, 300),
    (.98, 300, 300),
    (.7, 100, 300),
    (.8, 100, 300),
    (.9, 100, 300),
    (.98, 100, 300),
]
#params = None
experiment = ESNExperiment(LorenzSystem(), params=params)
res = experiment.run(verbose=True)
with open(fn, 'wb') as f:
    pickle.dump(res, f)
print(res)
