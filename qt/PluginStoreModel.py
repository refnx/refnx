from PySide import QtCore, QtGui
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
        return None, None
        
    return (module, rfos)
    
class PluginStoreModel(QtCore.QAbstractTableModel):
    def __init__(self, parent = None):
        super(PluginStoreModel, self).__init__(parent)
        self.plugins = []
        self.plugins.append({'name':'default', 'rfo':reflect.ReflectivityFitObject})
                
    def rowCount(self, parent = QtCore.QModelIndex()):
        return len(self.plugins)
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 1
            
    def insertRows(self, position, rows=1, index=QtCore.QModelIndex()):
        """ Insert a row into the model. """
        self.beginInsertRows(QtCore.QModelIndex(), position, position + rows - 1)
        self.endInsertRows()
        return True
        
    def flags(self, index):
        return  (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        self.dataChanged.emit(index, index)
        return True
                
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        
        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0:
                return self.plugins[index.row()]['name']
                
    def addPlugin(self, filepath):
        module, rfos = loadReflectivityModule(filepath)
        
        if rfos is None:
            return
            
        for obj in rfos:
            self.plugins.append({'name':obj[0], 'rfo':obj[1]})

        self.insertRows(len(self.plugins), rows = len(rfos))        
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())

        
class PluginParametersModel(QtCore.QAbstractTableModel):

    def __init__(self, model, parent = None):
        super(PluginParametersModel, self).__init__(parent)
        self.model = model
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        return len(self.model.parameters) + 1
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 1
    
    def flags(self, index):
        numlayers = int(self.model.parameters[0])
        row = index.row()
        col = index.column()
                    
    	return (QtCore.Qt.ItemIsEditable |
    	         QtCore.Qt.ItemIsUserCheckable |
    	           QtCore.Qt.ItemIsEnabled |
    	            QtCore.Qt.ItemIsSelectable)
    	            
    def layersAboutToBeInserted(self, start, end):
        self.beginInsertRows(QtCore.QModelIndex(), start, end)    
    
    def layersFinishedBeingInserted(self):
        self.endInsertRows()
        
    def layersAboutToBeRemoved(self, start, end):
        self.beginRemoveRows(QtCore.QModelIndex(), start, end)

    def layersFinishedBeingRemoved(self):
        self.endRemoveRows()
            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()
        
        if role == QtCore.Qt.CheckStateRole:
            fitted_parameters = self.model.fitted_parameters
            if value == QtCore.Qt.Checked:
                fitted_parameters = np.delete(fitted_parameters,np.where(fitted_parameters == row))
            else:
                fitted_parameters = np.append(fitted_parameters, param)
                
            self.model.fitted_parameters = fitted_parameters[:]
                
        if role == QtCore.Qt.EditRole:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(value, 1)
            if voutput[0] == QtGui.QValidator.State.Acceptable:
                self.model.parameters[param] = voutput[1]
            else:
                return False
        
        self.dataChanged.emit(index, index)
        return True
                
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False
        
        row = index.row()
        col = index.column()
            
        if role == QtCore.Qt.DisplayRole:
            if row == 0:
                return str(np.size(self.model.parameters))
            else:
                return str(self.model.parameters[row - 1])
        
        if role == QtCore.Qt.CheckStateRole:
            if param in self.model.fitted_parameters:
                return QtCore.Qt.Unchecked
            else:
               return QtCore.Qt.Checked
                