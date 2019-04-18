import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from echonn.sys import SystemSolver

res = pickle.load(open('rc3_body_results.p', 'rb'))
run, lce, ts_data, results = res

# print small details
print('lce:', lce[0])
print('ic:', run['results'].y[:, 0])
print('tf:', run['results'].t[-1])

rank_metric = np.array([test_rmse for _, cv_rmse, _, (_, _, test_rmse)
                        in results['best model rmse']])
ranking = np.argsort(rank_metric)


def rmse(d, y):
    num_samples = d.shape[0]
    return np.sqrt(np.sum((d-y)**2) / num_samples)


# get img dir
dir_pre = os.path.join('..', 'images', 'r3body')

# extract system and build solver
sys = run['system']
slvr = SystemSolver(sys)

# draw full data
runt = deepcopy(run)
runt['results'].y = run['results'].y[[0, 2], :]
slvr.plot2d(runt)
plt.scatter([-sys.alpha, sys.mu], [0, 0])
plt.savefig(os.path.join(dir_pre, 'full_differential.png'))

# draw test data
runt['results'].t = ts_data.test_t
runt['results'].y = ts_data.test_y[:, [0, 2]].T
slvr.plot2d(runt)
plt.scatter([-sys.alpha, sys.mu], [0, 0])
plt.savefig(os.path.join(dir_pre, 'test_data.png'))

how_many = 10
for rank, i in enumerate(ranking[:how_many]):
    ds_test, ys_test, total_rmse = results['best model rmse'][i][3]
    print(i, total_rmse, results['params'][i])

    rmse_over_t = [rmse(ds_test[:sub+1], ys_test[:sub+1])
                   for sub in range(ds_test.shape[0])]

    plt.figure()
    plt.title('RMSE vs Lyapunov Time\nParam {} ; RMSE {}'.format(
        results['params'][i], total_rmse))
    t_adj = (ts_data.test_t - ts_data.test_t[0]) * lce[0]
    plt.plot(t_adj, rmse_over_t, 'o-')
    plt.plot(t_adj, np.zeros_like(t_adj))
    name = 'rank_{}_param_{}_rmse.png'.format(rank, i)
    plt.savefig(os.path.join(dir_pre, name))
    plt.close()

    runt['results'].t = ts_data.test_t
    runt['results'].y = ys_test[:, [0, 2]].T
    slvr.plot2d(runt)
    plt.scatter([-sys.alpha, sys.mu], [0, 0])
    name = 'rank_{}_param_{}_fit.png'.format(rank, i)
    plt.savefig(os.path.join(
        dir_pre, name))
    plt.close()
    # plt.show(True)
