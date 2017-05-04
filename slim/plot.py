
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class SlimPlotWindow(QDialog):
    def __init__(self, parent=None):
        super(SlimPlotWindow, self).__init__(parent)

        # a figure instance to plot on
        self.figure = Figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some plot_button connected to `plot` method
        self.plot_button = QPushButton('Plot')
        self.plot_button.clicked.connect(self.plot)

        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.clicked.connect(self.refresh)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.refresh_button)
        self.setLayout(layout)

        self.files_displayed = []
        self.data_directory = '~/'

    def refresh(self):
        if self.files_displayed:
            self.plot(files_to_display=self.files_displayed)

    @pyqtSlot(str)
    def data_directory_changed(self, directory):
        self.data_directory = directory

    def plot(self, files_to_display=False):
        if not files_to_display:
            files_to_display = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                "Select reflectometry data files to plot",
                directory=self.data_directory,
                filter='Reflectometry files (*.xml)')

        if not files_to_display[0]:
            # could've cancelled the file dialogue
            return

        # load each file and display it
        TODO

        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()
