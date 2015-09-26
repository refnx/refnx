"""
Qt Model class for dealing with all the datasets.  It provides data for the
GUI. It deals with GUI requests for loading datasets as well.
"""
from __future__ import division
from PySide import QtCore, QtGui
from PySide.QtCore import QAbstractItemModel
import datastore
import numpy as np
import refnx.dataset.reflectdataset as reflectdataset


class DataStoreModel(QtCore.QAbstractTableModel):

    def __init__(self, parent=None):
        super(DataStoreModel, self).__init__(parent)
        self.datastore = datastore.DataStore()

    def __iter__(self):
        for dataset in self.datastore:
            yield dataset

    def __getitem__(self, key):
        return self.datastore[key]

    def __len__(self):
        return len(self.datastore)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self)

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
            return (QtCore.Qt.ItemIsUserCheckable |
                 QtCore.Qt.ItemIsEnabled |
                 QtCore.Qt.ItemIsSelectable)
        else:
            return (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if index.column() == 1:
            name = self.datastore.names[index.row()]
            dataset = self.datastore.datasets[name]

            if value == QtCore.Qt.Checked:
                dataset.graph_properties['visible'] = True
            else:
                dataset.graph_properties['visible'] = False

            self.dataChanged.emit(index, index)

        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None

        name = self.datastore.names[index.row()]
        dataset = self.datastore.datasets[name]

        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0:
                return name

        if role == QtCore.Qt.CheckStateRole:
            if index.column() == 1:
                if dataset.graph_properties['visible']:
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

    def add(self, dataset):
        self.datastore.add(dataset)
        self.insertRows(len(self.datastore))
        lindex = self.createIndex(len(self.datastore) - 1, 0)
        rindex = self.createIndex(len(self.datastore) - 1, 2)

        self.dataChanged.emit(lindex, rindex)

    def snapshot(self, snapshot_name):
        original = self.datastore['theoretical']
        dataset = reflectdataset.ReflectDataset()
        dataset.data = original.data
        dataset.y = np.copy(original.fit)
        dataset.name = snapshot_name
        self.add(dataset)
        return dataset

    def remove(self, name):
        index = self.datastore.names.index(name)
        self.datastore.remove_dataset(name)
        self.removeRows(index, 0)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def load(self, filename):
        dataset = self.datastore.load(filename)
        self.insertRows(len(self.datastore))
        lindex = self.createIndex(len(self.datastore) - 1, 0)
        rindex = self.createIndex(len(self.datastore) - 1, 2)

        self.dataChanged.emit(lindex, rindex)
        return dataset


