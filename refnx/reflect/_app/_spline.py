import os.path
import json

from qtpy import QtCore, QtGui, QtWidgets, uic

from refnx.reflect import Spline, SLD


pth = os.path.dirname(os.path.abspath(__file__))
UI_LOCATION = os.path.join(pth, "ui")


class SplineDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        # persistent lipid leaflet dlg
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi(os.path.join(UI_LOCATION, "spline.ui"), self)

        self.knots.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem("dz"))
        self.knots.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem("vs"))
        self.knots.setItem(0, 0, QtWidgets.QTableWidgetItem(str(0.5)))
        self.knots.setItem(0, 1, QtWidgets.QTableWidgetItem(str(-1)))
        self._dz_delegate = _dz_delegate()
        self._vs_delegate = _vs_delegate()
        self.knots.setItemDelegateForColumn(0, self._dz_delegate)
        self.knots.setItemDelegateForColumn(1, self._vs_delegate)

    @QtCore.Slot(int)
    def on_num_knots_valueChanged(self, val):
        oldrows = self.knots.rowCount()
        self.knots.setRowCount(val)
        for row in range(oldrows, val):
            self.knots.setItem(row, 0, QtWidgets.QTableWidgetItem(str(0.1)))
            self.knots.setItem(row, 1, QtWidgets.QTableWidgetItem(str(-1)))

    def component(self):
        # return a SplineComponent
        dz = []
        vs = []
        extent = self.extent.value()

        for i in range(self.knots.rowCount()):
            dz.append(float(self.knots.item(i, 0).text()))
            vs.append(float(self.knots.item(i, 1).text()))

        return Spline(
            extent, vs, dz, name="spline", microslab_max_thickness=1.0
        )


class _dz_delegate(QtWidgets.QItemDelegate):
    def createEditor(self, parent, option, index):
        d = QtWidgets.QDoubleSpinBox(parent)
        d.setRange(0, 1)
        d.setSingleStep(0.005)
        return d


class _vs_delegate(QtWidgets.QItemDelegate):
    def createEditor(self, parent, option, index):
        d = QtWidgets.QDoubleSpinBox(parent)
        d.setRange(-4, 150)
        d.setSingleStep(0.005)
        return d
