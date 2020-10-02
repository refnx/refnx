import pickle


def test_graphproperties():
    from matplotlib.figure import Figure
    from refnx.reflect._app.graphproperties import GraphProperties

    gp = GraphProperties()
    fig = Figure()
    axes = fig.add_axes([0.06, 0.15, 0.9, 0.8])
    ebc = axes.errorbar([0, 1, 2.0], [0, 1, 2.0], [0, 1, 2.0])
    gp.ax_data = ebc
    gp.save_graph_properties()

    pkl = pickle.dumps(gp)
    assert pkl is not None

    re_gp = pickle.loads(pkl)
    assert isinstance(re_gp, GraphProperties)

    assert re_gp.ax_data is None
