from __future__ import division
from PySide import QtCore, QtGui
import imp
import sys
import inspect
import hashlib
import os.path
import numpy as np
import refnx.analysis.reflect as reflect
from refnx.analysis import ReflectivityFitFunction
from refnx.analysis.curvefitter import CurveFitter
import refnx.analysis.curvefitter as curvefitter
from lmfit import Parameters, Parameter
from lmfit.astutils import valid_symbol_name
from collections import OrderedDict


def load_plugin_module(filepath):
    # this loads all modules
    hash = hashlib.md5(filepath)

    name = os.path.basename(filepath)
    name, ext = os.path.splitext(name)

    module = imp.load_source(name, filepath)

    rfos = []

    #load plugins that subclass CurveFitter
    members = inspect.getmembers(module, inspect.isclass)
    for member in members:
        if (member[1] == ReflectivityFitFunction
           or member[1] == CurveFitter):
           continue

        if issubclass(member[1], CurveFitter):
            rfos.append(member)
            print('Loaded', member[0], 'plugin fitting module')

    #also load functions that have the curvefitter.fitfunc decorator
    functions = inspect.getmembers(module, inspect.isfunction)
    for function in functions:
        if hasattr(function[1], 'fitfuncwraps'):
            rfos.append(function)

    if not len(rfos):
        del sys.modules[name]
        return None, None

    return (module, rfos)


class PluginStoreModel(QtCore.QAbstractTableModel):

    def __init__(self, parent=None):
        super(PluginStoreModel, self).__init__(parent)
        self.plugins = OrderedDict()
        self.plugins['default'] = (reflect.ReflectivityFitFunction, '')

    def __len__(self):
        return len(self.plugins)

    def __getitem__(self, key):
        return self.plugins[key][0]

    def __setitem__(self, key, value, filepath=''):
        if issubclass(value, curvefitter.CurveFitter):
            self.plugins[key] = (value, filepath)
        if hasattr(value, 'fitfuncwraps'):
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
        return 5

    def flags(self, index):
        row = index.row()
        col = index.column()

        if row == 0 and col == 0:
            retval = QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled
        
        if row == 0 and col > 0:
            retval = False

        #parameter name
        if col == 0 and row > 0:
            retval = (QtCore.Qt.ItemIsEditable |
                      QtCore.Qt.ItemIsEnabled |
                      QtCore.Qt.ItemIsSelectable)

        #parameter value
        if col == 1 and row > 0:
            retval = (QtCore.Qt.ItemIsEditable |
                      QtCore.Qt.ItemIsUserCheckable |
                      QtCore.Qt.ItemIsEnabled |
                      QtCore.Qt.ItemIsSelectable)

        #min/max values
        if (col == 2 or col == 3) and row > 0:
            retval = (QtCore.Qt.ItemIsEditable |
                      QtCore.Qt.ItemIsEnabled |
                      QtCore.Qt.ItemIsSelectable)

        #expr
        if col == 4 and row > 0:
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
            if row > 0 and col == 1:
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
                        for i in range(currentparams, newparams):
                            self.params.add('p%d'%i, 0, True, -np.inf, np.inf, None)
                        self.endInsertRows()

                    if newparams < currentparams:
                        remove_names = names[newparams:]
                        map(self.params.pop, remove_names)
                        self.endRemoveRows()

                    self.modelReset.emit()
            if row > 0 and col in [1, 2, 3]:
                validator = QtGui.QDoubleValidator()
                voutput = validator.validate(value, 1)
                if voutput[0] == QtGui.QValidator.State.Acceptable:
                    number = float(voutput[1])
                else:
                    return False

                if col == 1:
                    self.params[name].value = number
                if col == 2:
                    self.params[name].min = number
                if col == 3:
                    self.params[name].max = number
            if row > 0 and col == 0:
                #change a parameter name requires making a new dictionary
                if not valid_symbol_name(value):
                    return False

                p = Parameters()
                param = self.params[name]
                newparam = Parameter(value, param.value, param.vary,
                                     param.min, param.max, param.expr)

                for k, v in self.params.items():
                    if k == name:
                        p[value] = newparam
                    else:
                        p[k] = v

                self.params = p

            if row > 0 and col == 4:
                #set an expression
                param = self.params[name]
                param.expr = value

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
                    return name
            elif col == 1 and row > 0:
                    return str(self.params[name].value)
            elif col == 2 and row > 0:
                return str(self.params[name].min)
            elif col == 3 and row > 0:
                return str(self.params[name].max)
            elif col == 4 and row > 0:
                return str(self.params[name].expr)

        if role == QtCore.Qt.CheckStateRole:
            if row > 0 and col == 1:
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
                names = self.params.keys()
                return names[section - 1]

        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'name'
            if section == 1:
                return 'value'
            if section == 2:
                return 'lower limit'
            if section == 3:
                return 'upper limit'
            if section == 4:
                return 'expr'
        return None
