from __future__ import division
from PySide import QtCore, QtGui
import numpy as np
import refnx.analysis.reflect as reflect
import refnx.analysis.curvefitter as curvefitter


class BaseModel(QtCore.QAbstractTableModel):

    '''
        a model for displaying in a QtGui.QTableView
    '''

    layersAboutToBeInserted = QtCore.Signal(int, int)
    layersAboutToBeRemoved = QtCore.Signal(int, int)
    layersFinishedBeingInserted = QtCore.Signal()
    layersFinishedBeingRemoved = QtCore.Signal()

    def __init__(self, params, parent=None):
        super(BaseModel, self).__init__(parent)
        self.params = params

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
        row = index.row()
        col = index.column()

        coltopar = ['nlayers', 'scale', 'bkg']

        if row != 0 or col < 0 or col > 2:
            return False

        if role == QtCore.Qt.CheckStateRole and index.column() > 0:
            name = coltopar[col]

            if value == QtCore.Qt.Checked:
                self.params[name].vary = False
            else:
                self.params[name].vary = True

            return True

        if role == QtCore.Qt.EditRole:
            if col == 0:
                validator = QtGui.QIntValidator()
                voutput = validator.validate(value, 1)

                values = curvefitter.values(self.params)

                if not reflect.is_proper_Abeles_input(values):
                    raise ValueError('The size of the parameter array passed'
                                     ' to abeles should be 4 * coefs[0] + 8')

                if (voutput[0] is QtGui.QValidator.State.Acceptable
                    and int(voutput[1]) >= 0):

                    oldlayers = int(values[0])
                    newlayers = int(voutput[1])

                    if oldlayers == newlayers:
                        return True

                    if newlayers == 0:
                        start = 1
                        end = oldlayers
                        self.layersAboutToBeRemoved.emit(start, end)

                        names = curvefitter.names(self.params)[8:]
                        map(self.params.pop, names)
                        self.params['nlayers'].value = newlayers

                        thesignal = self.layersFinishedBeingRemoved
                    else:
                        if newlayers > oldlayers:
                            title = ('Where would you like to insert the new'
                                     ' layers')

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

                        self.params['nlayers'].value = newlayers
                        if newlayers > oldlayers:
                            start = insertpoint + 1
                            end = insertpoint + newlayers - oldlayers
                            self.layersAboutToBeInserted.emit(start, end)

                            values = curvefitter.values(self.params)
                            varys = curvefitter.varys(self.params)
                            bounds = curvefitter.bounds(self.params)

                            #do the insertion
                            startP = 4 * insertpoint + 8
                            endP = 4 * (insertpoint + newlayers
                                        - oldlayers) + 8
                            nvals = endP - startP

                            values = np.insert(values, startP, [0] * nvals)

                            dummy = [varys.insert(startP, i) for i
                                     in [True] * nvals]

                            dummy = [bounds.insert(startP, i) for i
                                     in [(None, None)] * nvals]

                            bounds = np.array(bounds)
                            names = reflect.parameter_names(values)

                            #clear the parameters
                            map(self.params.pop, self.params.keys())

                            # reinsert parameters
                            parlist = zip(names,
                                          values,
                                          varys,
                                          bounds.T[0],
                                          bounds.T[1],
                                          [None] * values.size)

                            for para in parlist:
                                self.params.add(*para)

                            thesignal = self.layersFinishedBeingInserted
                        elif newlayers < oldlayers:
                            start = insertpoint
                            end = insertpoint + (oldlayers - newlayers) - 1
                            self.layersAboutToBeRemoved.emit(start, end)

                            startP = 4 * (insertpoint - 1) + 8
                            endP = (4 * (insertpoint - 1 + oldlayers
                                         - newlayers) + 8)

                            #get rid of parameters we don't need anymore
                            names_lost = curvefitter.names(self.params)[startP: endP]
                            map(self.params.pop, names_lost)

                            # but now we need to rejig parameters names
                            # the only way to do this is to pop them all and readd
                            values = curvefitter.values(self.params)
                            varys = curvefitter.varys(self.params)
                            bounds = np.array(curvefitter.bounds(self.params))
                            names = reflect.parameter_names(values)
                            map(self.params.pop, self.params.keys())

                            parlist = zip(names,
                                          values,
                                          varys,
                                          bounds.T[0],
                                          bounds.T[1],
                                          [None] * values.size)

                            for para in parlist:
                                self.params.add(*para)


                            thesignal = self.layersFinishedBeingRemoved

                    # YOU HAVE TO RESIZE LAYER PARAMS
                    thesignal.emit()

                else:
                    return False
            else:
                validator = QtGui.QDoubleValidator()
                voutput = validator.validate(value, 1)
                if voutput[0] is QtGui.QValidator.State.Acceptable:
                    self.params[coltopar[index.column()]].value = voutput[1]
                else:
                    return False

        self.dataChanged.emit(index, index)
        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        values = curvefitter.values(self.params)

        if not reflect.is_proper_Abeles_input(values):
            return None

        if index.row() != 0 or index.column() < 0 or index.column() > 2:
            return None

        coltopar = ['nlayers', 'scale', 'bkg']

        if role == QtCore.Qt.DisplayRole:
            return str(self.params[coltopar[index.column()]].value)

        if role == QtCore.Qt.CheckStateRole:
            if (self.params[coltopar[index.column()]].vary
                and index.column() != 0):
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

    def __init__(self, params, parent=None):
        super(LayerModel, self).__init__(parent)
        self.params = params

    def rowCount(self, parent=QtCore.QModelIndex()):
        return int(self.params['nlayers'].value) + 2

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 4

    def flags(self, index):
        nlayers = int(self.params['nlayers'].value)
        row = index.row()
        col = index.column()

        if row == 0 and (col == 0 or col == 3):
            return QtCore.Qt.NoItemFlags

        if row == nlayers + 1 and col == 0:
            return QtCore.Qt.NoItemFlags

        return (QtCore.Qt.ItemIsEditable |
                QtCore.Qt.ItemIsUserCheckable |
                QtCore.Qt.ItemIsEnabled |
                QtCore.Qt.ItemIsSelectable)

    def rowcol_to_name(self, row, col, nlayers):
        if row == 0 and col == 1:
            param = 2
        elif row == 0 and col == 2:
            param = 3
        elif row == nlayers + 1 and col == 1:
            param = 4
        elif row == nlayers + 1 and col == 2:
            param = 5
        elif row == nlayers + 1 and col == 3:
            param = 7
        else:
            param = 4 * (row - 1) + col + 8

        names = curvefitter.names(self.params)
        return names[param]

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
        nlayers = int(self.params['nlayers'].value)

        if not index.isValid():
            return False

        if col < 0 or col > 3:
            return False

        name = self.rowcol_to_name(row, col, nlayers)

        if role == QtCore.Qt.CheckStateRole:
            if value == QtCore.Qt.Checked:
                self.params[name].vary = False
            else:
                self.params[name].vary = True

        if role == QtCore.Qt.EditRole:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(value, 1)
            if voutput[0] == QtGui.QValidator.State.Acceptable:
                self.params[name].value = float(voutput[1])
            else:
                return False

        self.dataChanged.emit(index, index)
        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False

        row = index.row()
        col = index.column()

        if not reflect.is_proper_Abeles_input(curvefitter.values(self.params)):
            return None

        nlayers = int(self.params['nlayers'].value)

        if row == 0 and (col == 0 or col == 3):
            return None

        if row == nlayers + 1 and col == 0:
            return None

        name = self.rowcol_to_name(row, col, nlayers)

        if role == QtCore.Qt.DisplayRole:
            return str(self.params[name].value)

        if role == QtCore.Qt.CheckStateRole:
            if self.params[name].vary:
                return QtCore.Qt.Unchecked
            else:
                return QtCore.Qt.Checked

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Vertical:
            nlayers = int(self.params['nlayers'].value)
            if section == 0:
                return 'fronting'
            elif section == nlayers + 1:
                return 'backing'
            else:
                return str(section)

        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'thickness'
            if section == 1:
                return 'SLD / 1e-6'
            if section == 2:
                return 'iSLD / 1e-6'
            if section == 3:
                return 'roughness'
        return None
