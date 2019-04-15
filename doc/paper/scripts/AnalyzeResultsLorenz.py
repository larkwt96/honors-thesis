import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from echonn.sys import SystemSolver

res = pickle.load(open('lorenz_results.p', 'rb'))
run, lce, ts_data, results = res

ds_test, ys_test, total_rmse = results['best model rmse'][0][3]

plt.figure()
plt.title('Param Config {} RMSE: {}'.format((.7, 500, 300), total_rmse))
slvr = SystemSolver(run['system'])
runt = deepcopy(run)
runt['results'].t = ts_data.test_t
runt['results'].y = ds_test.T
fig = slvr.plot3d(runt)
runt['results'].t = ts_data.test_t
runt['results'].y = ys_test.T
slvr.plot3d(runt, fig=fig)
plt.show(True)

# for i, rmse in enumerate(results['best model rmse']):
#test_rmse = rmse[3]
#ds_test, ys_test, total_rmse = test_rmse
#
# plt.figure()
#plt.title('Param Config {} RMSE: {}'.format(i, total_rmse))
#slvr = SystemSolver(run['system'])
#runt = deepcopy(run)
#runt['results'].t = ts_data.test_t
#runt['results'].y = ds_test.T
#fig = slvr.plot3d(runt)
#runt['results'].t = ts_data.test_t
#runt['results'].y = ys_test.T
#slvr.plot3d(runt, fig=fig)
# print()
#rerr = np.mean((ys_test - ds_test) / ds_test, axis=1)
# print(rerr)
# print(np.mean(rerr))
# print(total_rmse)

# plt.show(True)
#plt.plot(ts_data.test_t, ds_test)
#plt.plot(ts_data.test_t, np.mean((ds_test - ys_test) / ds_test, axis=1))
# plt.show(True)
