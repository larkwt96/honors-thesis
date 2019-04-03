from echonn.sys import CircleSystem, SystemSolver

if __name__ == "__main__":
    sys = SystemSolver(CircleSystem())
    lce, run = sys.get_lce()
    print('lce:', lce)
    print(run['results'].y[2:, -1].reshape(2, 2))
