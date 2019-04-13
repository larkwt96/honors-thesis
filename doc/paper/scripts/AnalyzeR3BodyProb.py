import os
import pickle
from echonn.sys import RestrictedCircular3Body
from echonn.ml import ESNExperiment


fn = os.path.join('.', 'rc_3_bod_results.p')
print('Testing Restricted Circular 3 Body Model')
experiment = ESNExperiment(RestrictedCircular3Body())
res = experiment.run()
with open(fn, 'wb') as f:
    pickle.dump(res, f)
print(res)
