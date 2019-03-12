import matplotlib.pyplot as plt


def clearFigs():
    # TODO: it doesn't want to close figures, and I don't know why. In the
    # mean time, I will just clear the figure, and the non blank figure is
    # the one you want.
    # return  # global disable
    for fignum in plt.get_fignums():
        plt.figure(fignum)
        plt.clf()
    """
    while len(plt.get_fignums()) > 0:
        fignum = plt.get_fignums()[0]
        plt.close(fignum)
        plt.pause(.001) # TODO: try this?
    """
