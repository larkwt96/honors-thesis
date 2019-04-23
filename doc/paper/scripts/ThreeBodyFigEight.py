import os
from echonn.sys import NBodySystem, SystemSolver
import numpy as np
import matplotlib.pyplot as plt

dir_pre = os.path.join('..', 'images', '3body')
sys = NBodySystem(body_masses=[1, 1, 1], G=1)
solver = SystemSolver(sys)
tspan = [0, 1.5]
y0 = np.zeros(2*sys.body_dim*len(sys.body_masses), dtype=np.float64)

x1 = np.array([0.97000436, -0.24308753, 0])
x3p = np.array([-0.93240737, -0.86473146, 0])

y0[0:3] = x1
y0[3:6] = -x1
y0[6:9] = 0
y0[9:12] = -x3p / 2
y0[12:15] = -x3p / 2
y0[15:18] = x3p
# print(sys.fun(np.zeros_like(y0), y0).reshape(6, -1))
lce, run = solver.get_lce(tspan[1], y0)
t = run['results'].t[1:]
y = run['results'].y[sys.dim:, 1:].reshape(sys.dim, sys.dim, -1)
#print(y[:, :, -1])
lces = []
for i, t_val in enumerate(t):
    Df_y = y[:, :, i]
    lces.append(solver.calc_lce(Df_y, t_val))
print(lces[-1])
print(np.mean(lces[-5:]))

plt.figure()
plt.title(f'Incomplete LCE ({lces[-1]:.2}) Convergence for 3 Body Figure 8')
plt.plot(t, lces)
plt.ylabel('LCE')
plt.ylabel('t')
plt.savefig(os.path.join(dir_pre, 'lce_converge.png'))

tspan = [0, 100]
# print(sys.fun(np.zeros_like(y0), y0).reshape(6, -1))
run = solver.run(tspan, y0)
# solver.plotnd(run)
# print(run['results'].y[:, -1].reshape(6, -1))
y_act = run['results'].y[:9]
run['results'].y = y_act[:3]
fig = solver.plot3d(run)
run['results'].y = y_act[3:6]
fig = solver.plot3d(run, fig=fig)
run['results'].y = y_act[6:9]
fig = solver.plot3d(run, fig=fig)
plt.title('Long Run of 3 Body Figure 8')
plt.tight_layout()
plt.savefig(os.path.join(dir_pre, 'fig8_long.png'))


tspan = [0, 10]
# print(sys.fun(np.zeros_like(y0), y0).reshape(6, -1))
run = solver.run(tspan, y0)
# solver.plotnd(run)
# print(run['results'].y[:, -1].reshape(6, -1))
y_act = run['results'].y[:9]
run['results'].y = y_act[:3]
fig = solver.plot3d(run)
run['results'].y = y_act[3:6]
fig = solver.plot3d(run, fig=fig)
run['results'].y = y_act[6:9]
fig = solver.plot3d(run, fig=fig)
plt.title('Short Run of 3 Body Figure 8')
plt.tight_layout()
plt.savefig(os.path.join(dir_pre, 'fig8_short.png'))
# plt.show(True)
