import matplotlib.pyplot as plt


def clearFigs():
    # this can fail, but it's not an issue.
    for fignum in plt.get_fignums():
        plt.figure(fignum)
        plt.clf()
