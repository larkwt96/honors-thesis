import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from echonn.sys import SystemSolver

res = pickle.load(open('rc3_body_results.p', 'rb'))
run, lce, ts_data, results = res
sys = run['system']
slvr = SystemSolver(sys)

# get img dir
dir_pre = os.path.join('..', 'images', 'r3body')

# print small details
print('lce:', lce[0])
# LCE
lce_val, lce_run = lce
T0 = 0
t = lce_run['results'].t[T0:]
y = lce_run['results'].y[:, T0:]
lces = []
for i, t_val in enumerate(t):
    Df_y0 = y[sys.dim:, i].reshape(sys.dim, sys.dim)
    lces.append(slvr.calc_lce(Df_y0, t_val))
plt.figure()
plt.title(f'Restricted 3 Body LCE ({lce_val:.3}) vs t')
plt.xlabel('t')
plt.ylabel('LCE')
plt.plot(t, lces)
plt.savefig(os.path.join(dir_pre, 'lce_converge.png'))
exit()

print('ic:', run['results'].y[:, 0])
print('tf:', run['results'].t[-1])

rank_metric_cv = np.array([cv_rmse for _, cv_rmse, _, (_, _, test_rmse)
                           in results['best model rmse']])
rank_metric_test = np.array([test_rmse for _, cv_rmse, _, (_, _, test_rmse)
                             in results['best model rmse']])
ranking = np.argsort(rank_metric_cv)


def rmse(d, y):
    num_samples = d.shape[0]
    return np.sqrt(np.sum((d-y)**2) / num_samples)


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

how_many = 5
for rank, i in enumerate(ranking[:how_many]):
    ds_test, ys_test, total_rmse = results['best model rmse'][i][3]
    print(i, total_rmse, results['params'][i])

    rmse_over_t = [rmse(ds_test[:sub+1], ys_test[:sub+1])
                   for sub in range(ds_test.shape[0])]

    plt.figure()
    plt.title('Test RMSE vs Lyapunov Time')
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
    title = 'ESN Trajectory\nParam {}\nValidation/Test RMSE {:.4}/{:.4}'
    param_title = '(α:{}, N:{}, T0:{})'.format(*results['params'][i])
    plt.title(title.format(param_title,
                           rank_metric_cv[i],
                           rank_metric_test[i]))
    name = 'rank_{}_param_{}_fit.png'.format(rank, i)
    plt.tight_layout()
    plt.savefig(os.path.join(
        dir_pre, name))
    plt.close()
    # plt.show(True)
