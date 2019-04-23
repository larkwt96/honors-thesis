import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from echonn.sys import SystemSolver

res = pickle.load(open('lorenz_results.p', 'rb'))
run, lce, ts_data, results = res
sys = run['system']
slvr = SystemSolver(sys)
dir_pre = os.path.join('..', 'images', 'lorenz')

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
plt.title(f'Lorenz LCE ({lce_val:.3}) vs t')
plt.xlabel('t')
plt.ylabel('LCE')
plt.plot(t, lces)
plt.savefig(os.path.join(dir_pre, 'lce_converge.png'))
exit()

# results
print(run['results'].y[:, 0])
test_rmse = [rmse for _, _, _, (_, _, rmse) in results['best model rmse']]
test_rmse = np.array(test_rmse)


def rmse(d, y):
    num_samples = d.shape[0]
    return np.sqrt(np.sum((d-y)**2) / num_samples)


score = []
higher_score = []
for rmse_res in results['best model rmse']:
    ds_test, ys_test, total_rmse = rmse_res[3]
    for sub in range(ds_test.shape[0]):
        err = rmse(ds_test[:sub+1], ys_test[:sub+1])
        if err > .05:
            score.append(sub)
            break
    for sub in range(ds_test.shape[0]):
        err = rmse(ds_test[:sub+1], ys_test[:sub+1])
        if err > 1:
            higher_score.append(sub)
            break
runt = deepcopy(run)
runt['results'].t = ts_data.t
runt['results'].y = ts_data.y.T
fig = slvr.plot3d(runt)
plt.savefig(os.path.join(dir_pre, 'full_differential.png'))

sorter = np.flip(np.argsort(score))
how_many = 8
for rank, i in enumerate(sorter[:how_many]):
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

    #runt['results'].t = ts_data.test_t
    #runt['results'].y = ds_test.T
    #fig = slvr.plot3d(runt)
    runt['results'].t = ts_data.test_t[:higher_score[i]+1]
    runt['results'].y = ds_test[:higher_score[i]+1].T
    fig = slvr.plot3d(runt)
    #runt['results'].t = ts_data.test_t[:score[i]+1]
    #runt['results'].y = ys_test[:score[i]+1].T
    #slvr.plot3d(runt, fig)
    runt['results'].t = ts_data.test_t[:higher_score[i]+1]
    runt['results'].y = ys_test[:higher_score[i]+1].T
    slvr.plot3d(runt, fig)
    plt.legend(['actual', 'approximation'])
    title = 'ESN Trajectory\nParam {}\nTest RMSE {:.4}'
    param_title = '(α:{}, N:{}, T0:{})'.format(*results['params'][i])
    plt.title(title.format(param_title, total_rmse))
    plt.tight_layout()
    name = 'rank_{}_param_{}_fit.png'.format(rank, i)
    plt.savefig(os.path.join(
        dir_pre, name))
    # plt.show(True)
    plt.close()
