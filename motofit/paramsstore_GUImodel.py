"""
Qt Model class for dealing with all the Parameters sets.  It provides data
for the GUI to display. It deals with GUI requests for loading models as well.
"""

from __future__ import division
from PySide import QtCore, QtGui
import datastore
import refnx.analysis.reflect as reflect
import refnx.analysis.curvefitter as curvefitter
from lmfit import Parameters

class ParamsStoreModel(QtCore.QAbstractListModel):

    def __init__(self, parent=None):
        super(ParamsStoreModel, self).__init__(parent)
        self.params_store = datastore.ParametersStore()

    def __iter__(self):
        for params in self.params_store:
            yield params

    def __getitem__(self, key):
        return self.params_store[key]

    def __setitem__(self, key, value):
        if isinstance(value, Parameters):
            self.params_store[key] = value

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.params_store)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return self.params_store.names[index.row()]

    def flags(self, index, filterNormalRef=True):
        names = self.params_store.names
        params = self.params_store[names[index.row()]]

        if self.params_store.displayOtherThanReflect:
            return (QtCore.Qt.ItemIsEnabled |
                    QtCore.Qt.ItemIsSelectable)
        else:
            values = curvefitter.values(params)
            if reflect.is_proper_abeles_input(values):
                return (QtCore.Qt.ItemIsEnabled |
                        QtCore.Qt.ItemIsSelectable)
            else:
                return (QtCore.Qt.NoItemFlags)

    def add(self, params, name):
        self.params_store.add(params, name)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())
