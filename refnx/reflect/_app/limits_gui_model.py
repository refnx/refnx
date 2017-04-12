from PyQt5 import QtCore, QtGui


class LimitsModel(QtCore.QAbstractTableModel):

    def __init__(self, model, finite_bounds=False, parent=None):
        super(LimitsModel, self).__init__(parent)
        self.model = model

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 5

    def rowCount(self, parent=QtCore.QModelIndex()):
        npars = len(self.model.parameters.varying_parameters())
        return npars

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False

        varying_params = self.model.parameters.varying_parameters()

        row = index.row()
        col = index.column()

        if role == QtCore.Qt.BackgroundRole and col < 2:
            return QtGui.QBrush(QtCore.Qt.cyan)

        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                return varying_params[row].name
            if col == 1:
                return str(varying_params[row].value)
            if col == 2:
                return str(varying_params[row].bounds.lb)
            if col == 3:
                return str(varying_params[row].bounds.ub)
            if col == 4:
                return str('')

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
        col = index.column()

        if col == 0 or col == 1:
            return QtCore.Qt.NoItemFlags

        return (QtCore.Qt.ItemIsEditable |
                QtCore.Qt.ItemIsEnabled |
                QtCore.Qt.ItemIsSelectable)

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()

        varying_params = self.model.parameters.varying_parameters()
        param = varying_params[row]

        if not index.isValid():
            return False

        if col not in [2, 3]:
            return False

        if role == QtCore.Qt.EditRole:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(value, 1)

            if voutput[0] == QtGui.QValidator.Acceptable:
                if col == 2:
                    param.bounds.lb = float(voutput[1])
                if col == 3:
                    param.bounds.ub = float(voutput[1])

        self.dataChanged.emit(index, index)
        return True
