import sys

from qtpy.compat import getopenfilenames
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QDialog, QPushButton, QVBoxLayout, QApplication
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

import numpy as np
from refnx.dataset import ReflectDataset


class SlimPlotWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.ax = None

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some plot_button connected to `plot` method
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.refresh_button)
        self.setLayout(layout)

        self.files_displayed = {}
        self.data_directory = "~/"

    def refresh(self):
        # refresh a dataset (may have been re-reduced)
        for k, v in self.files_displayed.items():
            dataset = v[0]
            line = v[1]
            dataset.refresh()
            self.adjustErrbarxy(line, *dataset.data[0:3])

        if self.ax is not None:
            # recompute the ax.dataLim
            self.ax.relim()
            # update ax.viewLim using the new dataLim
            self.ax.autoscale_view()

        self.canvas.draw()

    @QtCore.Slot(str)
    def data_directory_changed(self, directory):
        """
        This receives a signal from the main slim window to notify
        when the data directory was changed
        """
        self.data_directory = directory

    def plot(self, files_to_display=False):
        """
        Parameters
        ----------
        files_to_display : sequence of str
            filenames to display in the plot window
        """
        if not files_to_display:
            files = getopenfilenames(
                self,
                caption="Select reflectometry data files to plot",
                basedir=self.data_directory,
                filters="Reflectometry files (*.xml *.dat)",
            )
            files_to_display = files[0]

        if not files_to_display:
            # could've cancelled the file dialogue
            return

        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        self.ax = self.figure.add_subplot(111)

        displayed = {}

        # load each file and display it
        for file in files_to_display:
            dataset = ReflectDataset(file)

            # plot data
            line = self.ax.errorbar(*dataset.data[0:3], label=dataset.name)

            displayed[dataset.name] = (dataset, line)

        # add legend and plot log-lin
        self.ax.legend()
        self.ax.set_yscale("log")

        self.files_displayed = displayed

        # refresh canvas
        self.canvas.draw()

    def adjustErrbarxy(self, errobj, x, y, y_error):
        """for adjusting error bar plot with updated data"""

        # caplines and barsy have len > 1 if xerrors are displayed
        ln, caplines, barsy = errobj
        x_base = x
        y_base = y

        yerr_top = y_base + y_error
        yerr_bot = y_base - y_error

        new_segments_y = [
            np.array([[x, yt], [x, yb]])
            for x, yt, yb in zip(x_base, yerr_top, yerr_bot)
        ]
        barsy[0].set_segments(new_segments_y)
        ln.set_data(x, y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = SlimPlotWindow()
    ex.show()
    sys.exit(app.exec())
