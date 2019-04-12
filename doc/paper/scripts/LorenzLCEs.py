import numpy as np
from echonn.sys import LorenzSystem, SystemSolver

if __name__ == "__main__":
    data = [
        [16, 45.92, 4, 1.50255],
        [16, 40, 4, 1.37446],
        [10, 28, 8/3, 0.90566],
    ]
    results = []
    for sigma, rho, beta, lambda_ in data:

        lces = []
        print('calculating lces...')
        for i in range(10):
            sys = LorenzSystem(sigma, rho, beta)
            slv = SystemSolver(sys)

            lce, _ = slv.get_lce()
            print('\t{}:'.format(i), lce)
            lces.append(lce)
        res = {}
        res['beta'] = beta
        res['rho'] = rho
        res['sigma'] = sigma
        res['lambda'] = lambda_
        res['mean'] = np.mean(lces)
        res['std'] = np.std(lces)
        res['error'] = res['mean'] - lambda_
        res['relative error'] = res['error'] / lambda_
        print(res)
        print()
        results.append(res)

    print('results:')
    for res in results:
        toprint = [res['beta'], res['rho'], res['sigma'], res['lambda'],
                   res['mean'], res['relative error']]
        toprint = [str(v) for v in toprint]
        print('        ' + ' & '.join(toprint))
