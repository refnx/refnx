from refnx.dataset import ReflectDataset, Data1D


def refplot(datasets):
    """
    Quickly plot a lot of datasets

    Parameters
    ----------
    datasets : iterable
        {str, file, Data1D} specifying the datasets to plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure. Use fig.show() to display
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for dataset in datasets:
        if isinstance(dataset, Data1D):
            d = dataset
        else:
            d = ReflectDataset()
            d.load(dataset)
        ax.plot(d.x, d.y)

    ax.autoscale(tight=True)
    ax.set_yscale('log')
    ax.set_xlabel(u"Q /\u212B **-1")
    ax.set_ylabel('reflectivity')
    return fig
