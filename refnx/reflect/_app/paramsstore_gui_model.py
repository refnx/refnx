"""
Qt Model class for dealing with all the Parameters sets.  It provides data
for the GUI to display. It deals with GUI requests for loading models as well.
"""
from PyQt5 import QtCore, QtGui


class ParamsStoreModel(QtCore.QAbstractListModel):

    def __init__(self, datastore, parent=None):
        super(ParamsStoreModel, self).__init__(parent)
        self.datastore = datastore

    @property
    def models(self):
        models = [data_object.model for data_object in self.datastore
                  if data_object.model is not None]
        return models

    @property
    def model_names(self):
        names = [data_object.name for data_object in self.datastore
                 if data_object.model is not None]
        return names

    def __iter__(self):
        for model in self.models:
            yield model

    def __getitem__(self, key):
        return self.datastore[key].model

    def __setitem__(self, key, model):
        if model is not None:
            self.datastore[key].model = model

    def __len__(self):
        return len(self.models)

    def rowCount(self, parent=QtCore.QModelIndex()):
        # we don't need to display the theoretical model.
        return len(self) - 1

    def data(self, index, role=QtCore.Qt.DisplayRole):
        # we don't need to display the theoretical model.
        display_names = self.model_names
        display_names.remove('theoretical')

        if role == QtCore.Qt.DisplayRole:
            return display_names[index.row()]

    def flags(self, index, filterNormalRef=True):
        return (QtCore.Qt.ItemIsEnabled |
                QtCore.Qt.ItemIsSelectable)
