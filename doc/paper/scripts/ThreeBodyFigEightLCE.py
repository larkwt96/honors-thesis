import os
from echonn.sys import NBodySystem, SystemSolver
import numpy as np
import matplotlib.pyplot as plt

dir_pre = os.path.join('..', 'images', '3body')
sys = NBodySystem(body_masses=[1, 1, 1], body_dim=2, G=1)
solver = SystemSolver(sys)
tspan = [0, 200]
y0 = np.zeros(2*sys.body_dim*len(sys.body_masses), dtype=np.float64)

x1 = np.array([0.97000436, -0.24308753])
x3p = np.array([-0.93240737, -0.86473146])

y0[0:2] = x1
y0[2:4] = -x1
# y0[4:6] = zero
y0[6:8] = -x3p / 2
y0[8:10] = -x3p / 2
y0[10:12] = x3p

# print(sys.fun(np.zeros_like(y0), y0).reshape(6, -1))
lce, run = solver.get_lce(tspan[1], y0)
t = run['results'].t[1:]
y = run['results'].y[sys.dim:, 1:].reshape(sys.dim, sys.dim, -1)
#print(y[:, :, -1])
lces = []
for i, t_val in enumerate(t):
    Df_y = y[:, :, i]
    lces.append(solver.calc_lce(Df_y, t_val))

plt.figure()
plt.title(f'LCE ({lces[-1]:.2}) Convergence for 3 Body Figure 8')
plt.plot(t, lces)
plt.ylabel('LCE')
plt.xlabel('t')
plt.savefig(os.path.join(dir_pre, 'lce_converge.png'))
# plt.show(True)
