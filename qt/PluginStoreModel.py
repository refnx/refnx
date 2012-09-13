import imp
import sys
import inspect
import hashlib
import numpy as np
import pyplatypus.analysis.reflect as reflect

def loadReflectivityModule(filepath):
    #this loads all modules
    hash = hashlib.md5(filepath)
    module = imp.load_source(filepath, filepath)
    
    rfos = []
    
    members = inspect.getmembers(module, inspect.isclass)
    for member in members:
        if issubclass(member[1], reflect.ReflectivityFitObject):
            rfos.append(member)
    
    if not len(rfos):
        del sys.modules[filepath]
        return None
        
    return (module, rfos)
    
class PluginStoreModel(QtCore.QAbstractTableModel):
    def __init__(self, parent = None):
        super(PluginStoreModel, self).__init__(parent)
        self.plugins = []
                
    def rowCount(self, parent = QtCore.QModelIndex()):
        pass
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 1
            
    def insertRows(self, position, rows=1, index=QtCore.QModelIndex()):
        """ Insert a row into the model. """
        self.beginInsertRows(QtCore.QModelIndex(), position, position + rows - 1)
        self.endInsertRows()
        return True
        
    def flags(self, index):
        if index.column() == 1:
            return (QtCore.Qt.ItemIsUserCheckable |  QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        else:
            return  (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        if index.column() == 1:
            if value == QtCore.Qt.Checked:
                pass
            else:
                pass
                                
            self.dataChanged.emit(index, index)
        return True
                
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        
        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0:
                pass
        
        if role == QtCore.Qt.CheckStateRole:
             if index.column() == 1:
                if self.dataObjects[self.names[index.row()]].graph_properties['visible']:
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
        
        return None