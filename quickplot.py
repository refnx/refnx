from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

def the_ref_plot(name, xdata, ydata, yerr):
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.errorbar(xdata, ydata, yerr = yerr)
    ax.autoscale(tight=True)
    ax.loglog()
    ax.set_xlabel(u"Q /\u212B **-1")
    ax.set_ylabel('reflectivity')
    canvas = FigureCanvasAgg(fig)
    
    canvas.print_figure(name)
