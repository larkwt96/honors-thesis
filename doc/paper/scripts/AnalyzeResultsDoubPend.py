import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from echonn.sys import SystemSolver, DoublePendulumAnimator

res = pickle.load(open('doub_pend_results.p', 'rb'))
run, lce, ts_data, results = res

test_rmse = [rmse for _, _, _, (_, _, rmse) in results['best model rmse']]
test_rmse = np.array(test_rmse)


def rmse(d, y):
    num_samples = d.shape[0]
    return np.sqrt(np.sum((d-y)**2) / num_samples)


score = []
for rmse_res in results['best model rmse']:
    ds_test, ys_test, total_rmse = rmse_res[3]
    for sub in range(ds_test.shape[0]):
        err = rmse(ds_test[:sub+1], ys_test[:sub+1])
        if err > .2:
            score.append(sub)
            break

# doub_pend_anim = os.path.join('..', 'images', 'doub_pend_anim')
# anim = DoublePendulumAnimator(run)
# anim.render()
# anim.save(doub_pend_anim)

sorter = np.flip(np.argsort(score))
how_many = 10
for i in sorter[:how_many]:
    ds_test, ys_test, total_rmse = results['best model rmse'][i][3]
    print(i, total_rmse, results['params'][i])

    rmse_over_t = [rmse(ds_test[:sub+1], ys_test[:sub+1])
                   for sub in range(ds_test.shape[0])]

    plt.figure()
    plt.title('Param Config {} RMSE: {}'.format(
        results['params'][i], total_rmse))
    t_adj = (ts_data.test_t - ts_data.test_t[0]) / lce[0]
    plt.plot(t_adj, rmse_over_t, 'o-')
    plt.plot(t_adj, np.zeros_like(t_adj))

    slvr = SystemSolver(run['system'])
    runt = deepcopy(run)
    runt['results'].t = ts_data.test_t
    runt['results'].y = ds_test.T
    fig = slvr.plotnd(runt)
    runt['results'].t = ts_data.test_t
    runt['results'].y = ys_test.T
    slvr.plotnd(runt, fig)
    plt.title('Param Config {} RMSE: {}'.format(
        results['params'][i], total_rmse))
    plt.show(True)


def plot_one(ind):
    ds_test, ys_test, total_rmse = results['best model rmse'][ind][3]
    plt.show(True)


def plot_res():
    for i, rmse in enumerate(results['best model rmse']):
        test_rmse = rmse[3]
        ds_test, ys_test, total_rmse = test_rmse

        rerr = np.mean((ys_test - ds_test) / ds_test, axis=1)
        rerr = np.sqrt(rerr**2)

        plt.figure()
        plt.title('Param Config {} RMSE: {}'.format(
            results['params'][i], total_rmse))
        plt.plot((ts_data.test_t - ts_data.test_t[0]) / lce[0], rerr)

        slvr = SystemSolver(run['system'])
        runt = deepcopy(run)
        runt['results'].t = ts_data.test_t
        runt['results'].y = ds_test.T
        slvr.plotnd(runt)
        runt['results'].t = ts_data.test_t
        runt['results'].y = ys_test.T
        slvr.plotnd(runt)
        slvr.plot3d(runt)
        plt.title('Param Config {} RMSE: {}'.format(
            results['params'][i], total_rmse))
        plt.show(True)
