from __future__ import division
from PySide import QtCore, QtGui
import numpy as np
import pyplatypus.analysis.reflect as reflect
import pyplatypus.analysis.fitting as fitting


class BaseModel(QtCore.QAbstractTableModel):

    '''
        a model for displaying in a QtGui.QTableView
    '''

    layersAboutToBeInserted = QtCore.Signal(int, int)
    layersAboutToBeRemoved = QtCore.Signal(int, int)
    layersFinishedBeingInserted = QtCore.Signal()
    layersFinishedBeingRemoved = QtCore.Signal()

    def __init__(self, model, parent=None):
        super(BaseModel, self).__init__(parent)
        self.model = model

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 1

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 3

    def flags(self, index):
        return (QtCore.Qt.ItemIsEditable |
                QtCore.Qt.ItemIsUserCheckable |
                QtCore.Qt.ItemIsEnabled |
                QtCore.Qt.ItemIsSelectable)

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid():
            return False

        coltopar = [0, 1, 6]

        if index.row() != 0 or index.column() < 0 or index.column() > 2:
            return False

        if role == QtCore.Qt.CheckStateRole and index.column() > 0:
            fitted_parameters = np.copy(self.model.fitted_parameters)
            if value == QtCore.Qt.Checked:
                fitted_parameters = np.delete(
                    fitted_parameters,
                    np.where(fitted_parameters == coltopar[index.column()]))
            else:
                fitted_parameters = np.append(
                    fitted_parameters,
                    coltopar[index.column()])

            self.model.fitted_parameters = fitted_parameters[:]
            return True

        if role == QtCore.Qt.EditRole:
            if index.column() == 0:
                validator = QtGui.QIntValidator()
                voutput = validator.validate(value, 1)

                parameters = np.copy(self.model.parameters)

                if not reflect.is_proper_Abeles_input(parameters):
                    raise ValueError(
                        "The size of the parameter array passed to abeles should be 4 * coefs[0] + 8")

                fitted_parameters = np.copy(self.model.fitted_parameters)

                if voutput[0] is QtGui.QValidator.State.Acceptable and int(voutput[1]) >= 0:
                    oldlayers = int(parameters[0])
                    newlayers = int(voutput[1])
                    if oldlayers == newlayers:
                        return True

                    if newlayers == 0:
                        start = 1
                        end = oldlayers
                        self.layersAboutToBeRemoved.emit(start, end)

                        parameters.resize(8, refcheck=False)
                        thesignal = self.layersFinishedBeingRemoved
                        fitted_parameters = np.extract(
                            fitted_parameters < 8,
                            fitted_parameters)
                        parameters[0] = newlayers
                    else:
                        if newlayers > oldlayers:
                            title = 'Where would you like to insert the new layers'
                            maxValue = oldlayers
                            minValue = 0
                            value = 0
                        elif newlayers < oldlayers:
                            title = 'Where would you like to remove the layers from?'
                            maxValue = newlayers + 1
                            minValue = 1
                            value = 1

                        label = 'layer'
                        insertpoint, ok = QtGui.QInputDialog.getInt(None,
                                                                    title,
                                                                    label,
                                                                    value=value,
                                                                    minValue=minValue,
                                                                    maxValue=maxValue)
                        if not ok:
                            return False

                        parameters[0] = newlayers
                        if newlayers > oldlayers:
                            start = insertpoint + 1
                            end = insertpoint + newlayers - oldlayers
                            self.layersAboutToBeInserted.emit(start, end)

                            parameters = np.insert(parameters,
                                                   [4 * insertpoint + 8] *
                                                   4 *
                                                   (newlayers -
                                                    oldlayers),
                                                   [0, 0, 0, 0] * (newlayers - oldlayers))
                            fitted_parameters = np.where(
                                fitted_parameters >= 4 * insertpoint + 8,
                                fitted_parameters +
                                (newlayers - oldlayers) *
                                4,
                                fitted_parameters)
                            fitted_parameters = np.append(fitted_parameters,
                                                          np.arange(4 * insertpoint + 8, 4 * insertpoint + 8 + (newlayers - oldlayers) * 4))

                            thesignal = self.layersFinishedBeingInserted
                        elif newlayers < oldlayers:
                            insertpoint -= 1
                            start = insertpoint + 1
                            end = insertpoint + 1 + (oldlayers - newlayers) - 1
                            self.layersAboutToBeRemoved.emit(start, end)

                            paramslost = np.arange(
                                4 * insertpoint + 8,
                                4 * insertpoint + 8 + (oldlayers - newlayers) * 4)
                            parameters = np.delete(parameters, paramslost)
                            fitted_parameters = np.array(
                                [val for val in fitted_parameters.tolist() if (val < paramslost[0] or val > paramslost[-1])])
                            fitted_parameters = np.where(
                                fitted_parameters > paramslost[-1],
                                fitted_parameters +
                                (newlayers - oldlayers) * 4,
                                fitted_parameters)

                            thesignal = self.layersFinishedBeingRemoved

                    # YOU HAVE TO RESIZE LAYER PARAMS
                    self.model.parameters = parameters[:]
                    self.model.fitted_parameters = fitted_parameters[:]
                    thesignal.emit()

                else:
                    return False
            else:
                validator = QtGui.QDoubleValidator()
                voutput = validator.validate(value, 1)
                if voutput[0] is QtGui.QValidator.State.Acceptable:
                    self.model.parameters[
                        coltopar[index.column()]] = voutput[1]
                else:
                    return False

        self.dataChanged.emit(index, index)
        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None

        if not reflect.is_proper_Abeles_input(self.model.parameters):
            return None

        if index.row() != 0 or index.column() < 0 or index.column() > 2:
            return None

        coltopar = [0, 1, 6]

        if role == QtCore.Qt.DisplayRole:
            return str(self.model.parameters[coltopar[index.column()]])

        if role == QtCore.Qt.CheckStateRole:
            if coltopar[index.column()] in self.model.fitted_parameters and index.column() != 0:
                return QtCore.Qt.Unchecked
            else:
                return QtCore.Qt.Checked

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'layers'
            if section == 1:
                return 'scale factor'
            if section == 2:
                return 'background'
        return None


class LayerModel(QtCore.QAbstractTableModel):

    '''
        a model for displaying in a QtGui.QTableView
    '''

    def __init__(self, model, parent=None):
        super(LayerModel, self).__init__(parent)
        self.model = model

    def rowCount(self, parent=QtCore.QModelIndex()):
        return int(self.model.parameters[0]) + 2

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 4

    def flags(self, index):
        numlayers = int(self.model.parameters[0])
        row = index.row()
        col = index.column()

        if row == 0 and (col == 0 or col == 3):
            return QtCore.Qt.NoItemFlags

        if row == numlayers + 1 and col == 0:
            return QtCore.Qt.NoItemFlags

        return (QtCore.Qt.ItemIsEditable |
                QtCore.Qt.ItemIsUserCheckable |
                QtCore.Qt.ItemIsEnabled |
                QtCore.Qt.ItemIsSelectable)

    def rowcoltoparam(self, row, col, numlayers):
        if row == 0 and col == 1:
            param = 2
        elif row == 0 and col == 2:
            param = 3
        elif row == numlayers + 1 and col == 1:
            param = 4
        elif row == numlayers + 1 and col == 2:
            param = 5
        elif row == numlayers + 1 and col == 3:
            param = 7
        else:
            param = 4 * (row - 1) + col + 8

        return param

    def layersAboutToBeInserted(self, start, end):
        self.beginInsertRows(QtCore.QModelIndex(), start, end)

    def layersFinishedBeingInserted(self):
        self.endInsertRows()

    def layersAboutToBeRemoved(self, start, end):
        self.beginRemoveRows(QtCore.QModelIndex(), start, end)

    def layersFinishedBeingRemoved(self):
        self.endRemoveRows()

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()
        numlayers = int(self.model.parameters[0])

        if not index.isValid():
            return False

        if col < 0 or col > 3:
            return False

        param = self.rowcoltoparam(row, col, numlayers)

        if role == QtCore.Qt.CheckStateRole:
            fitted_parameters = self.model.fitted_parameters
            if value == QtCore.Qt.Checked:
                fitted_parameters = np.delete(
                    fitted_parameters,
                    np.where(fitted_parameters == param))
            else:
                fitted_parameters = np.append(fitted_parameters, param)

            self.model.fitted_parameters = fitted_parameters[:]

        if role == QtCore.Qt.EditRole:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(value, 1)
            if voutput[0] == QtGui.QValidator.State.Acceptable:
                self.model.parameters[param] = voutput[1]
            else:
                return False

        self.dataChanged.emit(index, index)
        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False

        row = index.row()
        col = index.column()

        if not reflect.is_proper_Abeles_input(self.model.parameters):
            return None

        numlayers = int(self.model.parameters[0])

        if row == 0 and (col == 0 or col == 3):
            return None

        if row == numlayers + 1 and col == 0:
            return None

        param = self.rowcoltoparam(row, col, numlayers)

        if role == QtCore.Qt.DisplayRole:
            return str(self.model.parameters[param])

        if role == QtCore.Qt.CheckStateRole:
            if param in self.model.fitted_parameters:
                return QtCore.Qt.Unchecked
            else:
                return QtCore.Qt.Checked

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Vertical:
            numlayers = (self.model.parameters[0])
            if section == 0:
                return 'fronting'
            elif section == numlayers + 1:
                return 'backing'
            else:
                return str(section)

        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'thickness'
            if section == 1:
                return 'SLD / 1e-6'
            if section == 2:
                return 'iSLD'
            if section == 3:
                return 'roughness'
        return None
