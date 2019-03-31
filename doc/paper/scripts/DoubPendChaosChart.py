import numpy as np
import os
from echonn.sys import DoublePendulumSystem, SystemSolver
import matplotlib.pyplot as plt
from scipy.constants import pi

if __name__ == "__main__":
    sys = DoublePendulumSystem()
    slv = SystemSolver(sys)
    thetas = np.arange(0.001, pi+0.0001, pi/20)
    out_img = os.path.join('..', 'images', 'chaos_vs_energy_in_doub_pend.png')
    lces = []
    for theta in thetas:
        lce, _ = slv.get_lce(T=200, y0=[theta, 0, 0, 0])
        lces.append(lce)
    plt.title('Largest LCE vs Inner Theta IC')
    plt.xlabel('Inner Theta')
    plt.ylabel('Largest LCE')
    plt.plot(180*thetas/pi, lces)
    try:
        if os.path.exists(out_img):
            os.remove(out_img)
    except:
        pass  # don't worry about it
    plt.savefig(out_img)
    plt.show(True)
