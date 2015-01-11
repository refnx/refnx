from __future__ import division
from PySide import QtCore, QtGui
import numpy as np
import refnx.analysis.curvefitter as curvefitter


class LimitsModel(QtCore.QAbstractTableModel):

    def __init__(self, params, finite_bounds=False, parent=None):
        super(LimitsModel, self).__init__(parent)
        self.params = params
        self.finite_bounds = finite_bounds
        self.varys = curvefitter.varys(self.params)
        self.fitted_params = np.where(self.varys)[0]
        self.names = curvefitter.names(self.params)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 5

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.params)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False

        row = index.row()
        col = index.column()

        name = self.names[row]
        param = self.params[name]

        if role == QtCore.Qt.BackgroundRole and col < 2:
            return QtGui.QBrush(QtCore.Qt.cyan)

        if role == QtCore.Qt.BackgroundRole and (not row in self.fitted_params
                                                 and col != 4):
            return QtGui.QBrush(QtCore.Qt.cyan)

        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                return name
            if col == 1:
                return str(param.value)
            if col == 2:
                return str(param.min)
            if col == 3:
                return str(param.max)
            if col == 4:
                return str(param.expr)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Vertical:
            return ''
        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'parameter'
            if section == 1:
                return 'value'
            if section == 2:
                return 'lower limit'
            if section == 3:
                return 'upper limit'
            if section == 4:
                return 'expr'
        return None

    def flags(self, index):
        row = index.row()
        col = index.column()

        if col == 0 or col == 1:
            return QtCore.Qt.NoItemFlags

        if (col == 2 or col == 3) and (not row in self.fitted_params):
            return QtCore.Qt.NoItemFlags

        return (QtCore.Qt.ItemIsEditable |
                QtCore.Qt.ItemIsEnabled |
                QtCore.Qt.ItemIsSelectable)

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()

        if not index.isValid():
            return False

        if col < 2 or col > 4:
            return False

        if not row in self.fitted_params and col < 4:
            return False

        name = self.names[row]
        param = self.params[name]

        none = ['none', 'None']
        lowlim = ['none', 'None', '-inf', '-Inf']
        hilim = ['none', 'None', 'inf', 'Inf']

        if role == QtCore.Qt.EditRole:
            if col == 4:
                if value in none :
                    value = None
                param.expr = value

            if self.finite_bounds:
                validator = QtGui.QDoubleValidator()
                voutput = validator.validate(value, 1)
                val = param.value

                if voutput[0] == QtGui.QValidator.State.Acceptable:
                    if col == 2:
                        param.min = float(voutput[1])
                        param.value = val
                    if col == 3:
                        param.max = float(voutput[1])
                        param.value = val
            else:
                if col == 2:
                    if value in lowlim:
                        param.min = -np.inf
                if col == 3:
                    if value in hilim:
                        param.max = np.inf

        self.dataChanged.emit(index, index)
        return True
