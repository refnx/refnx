from __future__ import division
from PySide import QtCore, QtGui
import datastore
import numpy as np
from dataobject import DataObject


class DataStoreModel(QtCore.QAbstractTableModel):

    def __init__(self, parent=None):
        super(DataStoreModel, self).__init__(parent)
        self.dataStore = datastore.DataStore()

    def __iter__(self):
        for dataObject in self.dataStore:
            yield dataObject

    def rowCount(self, parent=QtCore.QModelIndex()):
        return self.dataStore.numDataObjects

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 3

    def insertRows(self, position, rows=1, index=QtCore.QModelIndex()):
        """ Insert a row into the model. """
        self.beginInsertRows(
            QtCore.QModelIndex(),
            position,
            position + rows - 1)
        self.endInsertRows()
        return True

    def removeRows(self, row, count):
        self.beginRemoveRows(QtCore.QModelIndex(), row, row + count)
        self.endRemoveRows()

    def flags(self, index):
        if index.column() == 1:
            return (
                (QtCore.Qt.ItemIsUserCheckable |
                 QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            )
        else:
            return (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if index.column() == 1:
            if value == QtCore.Qt.Checked:
                self.dataStore.dataObjects[
                    self.dataStore.names[
                        index.row(
                        )]].graph_properties[
                    'visible'] = True
            else:
                self.dataStore.dataObjects[
                    self.dataStore.names[
                        index.row(
                        )]].graph_properties[
                    'visible'] = False

            self.dataChanged.emit(index, index)

        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0:
                return self.dataStore.names[index.row()]

        if role == QtCore.Qt.CheckStateRole:
            if index.column() == 1:
                if self.dataStore.dataObjects[self.dataStore.names[index.row()]].graph_properties['visible']:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'name'
            if section == 1:
                return 'displayed'
            if section == 2:
                return 'offset'

        return None

    def add(self, dataObject):
        self.dataStore.add(dataObject)
        self.insertRows(self.dataStore.numDataObjects)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def snapshot(self, snapshotname):
        original = self.dataStore['theoretical']
        dataTuple = (np.copy(original.xdata), np.copy(original.fit))
        snapshot = DataObject(name=snapshotname, dataTuple=dataTuple)
        self.add(snapshot)
        self.insertRows(self.dataStore.numDataObjects)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def remove(self, name):
        index = self.dataStore.names.index(name)
        self.dataStore.remove_DataObject(name)
        self.removeRows(index, 0)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def load(self, file):
        dataObject = self.dataStore.load(file)
        self.insertRows(self.dataStore.numDataObjects)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())
        return dataObject
