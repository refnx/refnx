from PySide import QtCore, QtGui
import imp
import sys
import inspect
import hashlib
import numpy as np
import os.path
import pyplatypus.analysis.reflect as reflect
import pyplatypus.analysis.fitting as fitting

def loadPluginModule(filepath):
    #this loads all modules
    hash = hashlib.md5(filepath)
    
    name = os.path.basename(filepath)
    name,ext=os.path.splitext(name)
    
    module = imp.load_source(name, filepath)
    
    rfos = []
    
    members = inspect.getmembers(module, inspect.isclass)
    for member in members:
        if issubclass(member[1], fitting.FitObject):
            rfos.append(member)
            print 'Loaded', name, 'plugin fitting module'
    
    if not len(rfos):
        del sys.modules[name]
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
        module, rfos = loadPluginModule(filepath)
        
        if rfos is None:
            return
            
        for obj in rfos:
            self.plugins.append({'name':obj[0], 'rfo':obj[1], 'filename' : filepath})

        self.insertRows(len(self.plugins), rows = len(rfos))        
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())

        
class UDFParametersModel(QtCore.QAbstractTableModel):

    def __init__(self, model, parent = None):
        super(UDFParametersModel, self).__init__(parent)
        
        self.model = model
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        return len(self.model.parameters) + 1
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 3
    
    def flags(self, index):
        numlayers = int(self.model.parameters[0])
        row = index.row()
        col = index.column()
                        
        if row == 0:
            retval = QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled

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
            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()
        
        if role == QtCore.Qt.CheckStateRole:
            if row > 0 and col == 0:
                fitted_parameters = self.model.fitted_parameters
                if value == QtCore.Qt.Checked:
                    fitted_parameters = np.delete(fitted_parameters,np.where(fitted_parameters == row - 1))
                else:
                    fitted_parameters = np.append(fitted_parameters, row - 1)
                
                self.model.fitted_parameters = fitted_parameters[:]
                
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
                    self.beginInsertRows(QtCore.QModelIndex(), currentparams + 1, newparams)
                if newparams < currentparams:
                     self.beginRemoveRows(QtCore.QModelIndex(), newparams + 1, currentparams)

                self.model.parameters = np.resize(self.model.parameters, newparams)
                self.model.parameters[currentparams:-1] = 0.

                if newparams > currentparams:
                    self.model.limits = np.append(self.model.limits, np.zeros((2, newparams - currentparams)),axis = 1)
                    self.model.limits[:, currentparams:-1] = 0.
                    self.model.fitted_parameters = np.append(self.model.fitted_parameters, range(currentparams, newparams + 1))
                    self.endInsertRows()
                if newparams < currentparams:                
                    self.model.limits = self.model.limits[:, 0: newparams]
                    self.endRemoveRows()
                        
            if row > 0:
                validator = QtGui.QDoubleValidator()
                voutput = validator.validate(value, 1)
                if voutput[0] == QtGui.QValidator.State.Acceptable:
                    number = voutput[1]
                else:
                    return False
                
                if col == 0:
                    self.model.parameters[row - 1] = number
                if col == 1:
                    self.model.limits[0, row - 1] = number
                if col == 2:
                    self.model.limits[1, row - 1] = number
                    
                
        
        self.dataChanged.emit(index, index)
        return True
                
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False
        
        row = index.row()
        col = index.column()
            
        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                if row == 0:
                    return str(np.size(self.model.parameters))
                else:
                    return str(self.model.parameters[row - 1])
            elif col == 1 and row > 0:
                return str(self.model.limits[0, row - 1])
            elif col == 2 and row > 0:
                return str(self.model.limits[1, row - 1])
                
        if role == QtCore.Qt.CheckStateRole:
            if row > 0 and col == 0:
                if (row - 1) in self.model.fitted_parameters:
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
                return str(section)
            
        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'value'
            if section == 1:
                return 'lower limit'
            if section == 2:
                return 'upper limit'
        return None
                