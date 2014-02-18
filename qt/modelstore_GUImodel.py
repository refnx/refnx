from __future__ import division
from PySide import QtCore, QtGui
import datastore
import pyplatypus.analysis.reflect as reflect


class ModelStoreModel(QtCore.QAbstractListModel):
    def __init__(self, parent = None):
        super(ModelStoreModel, self).__init__(parent)
        self.modelStore = datastore.ModelStore() 

    def __iter__(self):
        models = [self.modelStore[name] for name in self.names]
        for model in models:
            yield model
                    
    def rowCount(self, parent = QtCore.QModelIndex()):
        return len(self.modelStore.models)
        
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return self.modelStore.names[index.row()]
            
    def flags(self, index, filterNormalRef = True):
        parameters = self.modelStore.models[self.modelStore.names[index.row()]].parameters
        if self.modelStore.displayOtherThanReflect:
        	return (QtCore.Qt.ItemIsEnabled |
    	            QtCore.Qt.ItemIsSelectable)
        else:
            if reflect.is_proper_Abeles_input(parameters):
            	return (QtCore.Qt.ItemIsEnabled |
        	            QtCore.Qt.ItemIsSelectable)
            else:
            	return (QtCore.Qt.NoItemFlags)
        
    def add(self, model, modelName):
        self.modelStore.add(model, modelName)
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())        