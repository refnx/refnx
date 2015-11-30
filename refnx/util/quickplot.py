import matplotlib.pyplot as plt
from refnx.dataset import ReflectDataset

def ref_plot(datasets):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for dataset in datasets:
        d = ReflectDataset()
        d.load(dataset)
        ax.plot(d.x, d.y)

    ax.autoscale(tight=True)
    ax.set_yscale('log')
    ax.set_xlabel(u"Q /\u212B **-1")
    ax.set_ylabel('reflectivity')
    return fig