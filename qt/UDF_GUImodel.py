from __future__ import division
from PySide import QtCore, QtGui
import imp
import sys
import inspect
import hashlib
import os.path
import numpy as np
import refnx.analysis.reflect as reflect
from refnx.analysis.curvefitter import CurveFitter
import refnx.analysis.curvefitter as curvefitter
from lmfit import Parameters
from collections import OrderedDict


def load_plugin_module(filepath):
    # this loads all modules
    hash = hashlib.md5(filepath)

    name = os.path.basename(filepath)
    name, ext = os.path.splitext(name)

    module = imp.load_source(name, filepath)

    rfos = []

    members = inspect.getmembers(module, inspect.isclass)
    for member in members:
        if issubclass(member[1], CurveFitter):
            rfos.append(member)
            print 'Loaded', name, 'plugin fitting module'

    if not len(rfos):
        del sys.modules[name]
        return None, None

    return (module, rfos)


class PluginStoreModel(QtCore.QAbstractTableModel):

    def __init__(self, parent=None):
        super(PluginStoreModel, self).__init__(parent)
        self.plugins = OrderedDict()
        self.plugins['default'] = (reflect.ReflectivityFitter, '')

    def __len__(self):
        return len(self.plugins)

    def __getitem__(self, key):
        return self.plugins[key][0]

    def __setitem__(self, key, value, filepath=''):
        if issubclass(value, curvefitter.CurveFitter):
            self.plugins[key] = (value, filepath)

    @property
    def names(self):
        return list(self.plugins.keys())

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 1

    def insertRows(self, position, rows=1, index=QtCore.QModelIndex()):
        """ Insert a row into the model. """
        self.beginInsertRows(
            QtCore.QModelIndex(),
            position,
            position + rows - 1)
        self.endInsertRows()
        return True

    def flags(self, index):
        return (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        self.dataChanged.emit(index, index)
        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0:
                return self.names[index.row()]

    def add(self, filepath):
        module, rfos = load_plugin_module(filepath)

        if rfos is None:
            return

        for obj in rfos:
            self.plugins[obj[0]] = (obj[1], filepath)

        self.insertRows(len(self.plugins), rows=len(rfos))
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())


class UDFParametersModel(QtCore.QAbstractTableModel):

    def __init__(self, params, parent=None):
        super(UDFParametersModel, self).__init__(parent)
        if params is not None:
            self.params = params
        else:
            self.params = Parameters()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.params) + 1

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 3

    def flags(self, index):
        row = index.row()
        col = index.column()

        if row == 0 and col == 0:
            retval = QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled
        
        if row == 0 and col > 0:
            retval = False
            
        if col == 0 and row > 0:
            retval = (QtCore.Qt.ItemIsEditable |
                      QtCore.Qt.ItemIsUserCheckable |
                      QtCore.Qt.ItemIsEnabled |
                      QtCore.Qt.ItemIsSelectable)

        if col > 0 and row > 0:
            retval = (QtCore.Qt.ItemIsEditable |
                      QtCore.Qt.ItemIsEnabled |
                      QtCore.Qt.ItemIsSelectable)
        return retval

 #    def layersAboutToBeInserted(self, start, end):
#         self.beginInsertRows(QtCore.QModelIndex(), start, end)
#
#     def layersFinishedBeingInserted(self):
#         self.endInsertRows()
#
#     def layersAboutToBeRemoved(self, start, end):
#         self.beginRemoveRows(QtCore.QModelIndex(), start, end)
#
#     def layersFinishedBeingRemoved(self):
#         self.endRemoveRows()

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()
        names = curvefitter.names(self.params)

        if row:
            name = names[row - 1]

        if role == QtCore.Qt.CheckStateRole:
            if row > 0 and col == 0:
                if value == QtCore.Qt.Checked:
                    self.params[name].vary = False
                else:
                    self.params[name].vary = True

        if role == QtCore.Qt.EditRole:
            if row == 0 and col == 0:
                currentparams = self.rowCount() - 1

                validator = QtGui.QIntValidator()
                voutput = validator.validate(value, 1)
                if voutput[0] is QtGui.QValidator.State.Acceptable and int(voutput[1]) >= 0:
                    newparams = int(voutput[1])

                    if newparams == currentparams:
                        return True

                    if newparams > currentparams:
                        self.beginInsertRows(
                            QtCore.QModelIndex(),
                            currentparams + 1,
                            newparams)
                    if newparams < currentparams:
                        self.beginRemoveRows(
                            QtCore.QModelIndex(),
                            newparams + 1,
                            currentparams)

                    if newparams > currentparams:
                        self.model.limits = np.append(
                            self.model.limits, np.zeros((2, newparams - currentparams)), axis=1)
                        defaultlimits = self.model.default_limits()
                        self.model.limits[
                            :,
                            currentparams:-
                            1] = defaultlimits[
                            :,
                            currentparams:-
                            1]
                        self.model.fitted_parameters = np.append(
                            self.model.fitted_parameters, range(currentparams, newparams))
                        self.endInsertRows()

                    if newparams < currentparams:
                        remove_names = names[newparams:]
                        map(self.params.pop, remove_names)
                        self.endRemoveRows()

                    self.modelReset.emit()
            if row > 0:
                validator = QtGui.QDoubleValidator()
                voutput = validator.validate(value, 1)
                if voutput[0] == QtGui.QValidator.State.Acceptable:
                    number = voutput[1]
                else:
                    print 'not true'
                    return False

                if col == 0:
                    self.params[name].value = number
                if col == 1:
                    self.params[name].min = number
                if col == 2:
                    self.params[name].max = number

        self.dataChanged.emit(index, index)
        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False

        row = index.row()
        col = index.column()
        names = curvefitter.names(self.params)

        if row:
            name = names[row - 1]

        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                if row == 0:
                    return str(len(self.params))
                else:
                    return str(self.params[name].value)

            elif col == 1 and row > 0:
                return str(self.params[name].min)
            elif col == 2 and row > 0:
                return str(self.params[name].max)

        if role == QtCore.Qt.CheckStateRole:
            if row > 0 and col == 0:
                if self.params[name].vary:
                    return QtCore.Qt.Unchecked
                else:
                    return QtCore.Qt.Checked

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Vertical:
            if section == 0:
                return 'number of parameters'
            else:
                return str(section - 1)

        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'value'
            if section == 1:
                return 'lower limit'
            if section == 2:
                return 'upper limit'
        return None
