"""
Qt Model class for dealing with all the datasets.  It provides data for the
GUI. It deals with GUI requests for loading datasets as well.
"""
from copy import deepcopy

from PyQt5 import QtCore
from refnx.dataset import ReflectDataset


class DataStoreModel(QtCore.QAbstractTableModel):

    def __init__(self, datastore, parent=None):
        super(DataStoreModel, self).__init__(parent)
        self.datastore = datastore

    def __iter__(self):
        for data_object in self.datastore:
            yield data_object

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
            dataset = self.datastore.data_objects[name]

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
        dataset = self.datastore.data_objects[name]

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

    def snapshot(self, snapshot_name):
        original = self.datastore['theoretical']
        dataset = ReflectDataset()
        dataset.data = (original.dataset.x,
                        original.model.model(original.dataset.x,
                                             x_err=dataset.x_err))
        dataset.name = snapshot_name

        # if the snapshot already exists then overwrite it.
        data_object = self[dataset.name]
        if data_object is not None:
            data_object.dataset = dataset
        else:
            data_object = self.datastore.add(dataset)
            self.modelReset.emit()

        # associate a model with the snapshot
        new_model = deepcopy(original.model)
        new_model.name = snapshot_name
        data_object.model = new_model

        return data_object

    def remove(self, name):
        index = self.datastore.names.index(name)
        self.datastore.remove_dataset(name)
        self.removeRows(index, 0)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())
