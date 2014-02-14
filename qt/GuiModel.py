from __future__ import division
from PySide import QtCore, QtGui
import DataStore
import DataObject
import imp
import sys
import inspect
import hashlib
import os.path
import numpy as np
import pyplatypus.analysis.reflect as reflect
import pyplatypus.analysis.fitting as fitting


class DataStoreModel(QtCore.QAbstractTableModel):

    def __init__(self, parent = None):
        super(DataStoreModel, self).__init__(parent)
        self.dataStore = DataStore.DataStore()
        
    def __iter__(self):
        for dataObject in self.dataStore:
            yield dataObject
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        return self.dataStore.numDataObjects
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 3
    
    def insertRows(self, position, rows=1, index=QtCore.QModelIndex()):
        """ Insert a row into the model. """
        self.beginInsertRows(QtCore.QModelIndex(), position, position + rows - 1)
        self.endInsertRows()
        return True

        
    def removeRows(self, row, count):
        self.beginRemoveRows(QtCore.QModelIndex(), row, row + count)
        self.endRemoveRows()
                
    def flags(self, index):
        if index.column() == 1:
            return (QtCore.Qt.ItemIsUserCheckable |  QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        else:
            return  (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        if index.column() == 1:
            if value == QtCore.Qt.Checked:
                self.dataStore.dataObjects[self.dataStore.names[index.row()]].graph_properties['visible'] = True
            else:
                self.dataStore.dataObjects[self.dataStore.names[index.row()]].graph_properties['visible'] = False
                                
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
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())
    
    def snapshot(self, snapshotname):        
        original = self.dataStore['theoretical']
        dataTuple = (np.copy(original.W_q), np.copy(original.fit))
        snapshot = DataObject.DataObject(name = snapshotname, dataTuple = dataTuple)
        self.add(snapshot)
        self.insertRows(self.dataStore.numDataObjects)
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())
        
    def remove(self, name):
        index = self.dataStore.names.index(name)
        self.dataStore.removeDataObject(name)
        self.removeRows(index, 0)
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())
        
    def load(self, file):
        dataObject = self.dataStore.load(file)
        self.insertRows(self.dataStore.numDataObjects)
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())
        return dataObject

class ModelStoreModel(QtCore.QAbstractListModel):
    def __init__(self, parent = None):
        super(ModelStoreModel, self).__init__(parent)
        self.modelStore = DataStore.ModelStore() 

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
            if reflect.isProperAbelesInput(parameters):
            	return (QtCore.Qt.ItemIsEnabled |
        	            QtCore.Qt.ItemIsSelectable)
            else:
            	return (QtCore.Qt.NoItemFlags)
        
    def add(self, model, modelName):
        self.modelStore.add(model, modelName)
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())            
 
class BaseModel(QtCore.QAbstractTableModel):

    layersAboutToBeInserted = QtCore.Signal(int, int)
    layersAboutToBeRemoved = QtCore.Signal(int, int)
    layersFinishedBeingInserted = QtCore.Signal()
    layersFinishedBeingRemoved = QtCore.Signal()
    
    def __init__(self, model, parent = None):
        super(BaseModel, self).__init__(parent)
        self.model = model
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        return 1
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 3
    
    def flags(self, index):
    	return (QtCore.Qt.ItemIsEditable |
    	         QtCore.Qt.ItemIsUserCheckable |
    	           QtCore.Qt.ItemIsEnabled |
    	            QtCore.Qt.ItemIsSelectable)
            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        if not index.isValid():
            return False
        
        coltopar = [0, 1, 6]
        
        if index.row() != 0 or index.column() < 0 or index.column() > 2:
            return False

        if role == QtCore.Qt.CheckStateRole and index.column() > 0:
            fitted_parameters = np.copy(self.model.fitted_parameters)
            if value == QtCore.Qt.Checked:
                fitted_parameters = np.delete(fitted_parameters,np.where(fitted_parameters == coltopar[index.column()]))
            else:
                fitted_parameters = np.append(fitted_parameters, coltopar[index.column()])
                
            self.model.fitted_parameters = fitted_parameters[:]
            return True
            
        if role == QtCore.Qt.EditRole:
            if index.column() == 0:
                validator = QtGui.QIntValidator()
                voutput = validator.validate(value, 1)
                
                parameters = np.copy(self.model.parameters)
                
                if not reflect.isProperAbelesInput(parameters):
                    raise ValueError("The size of the parameter array passed to abeles should be 4 * coefs[0] + 8")
                
                fitted_parameters = np.copy(self.model.fitted_parameters)
                
                if voutput[0] is QtGui.QValidator.State.Acceptable and int(voutput[1]) >= 0:
                    oldlayers = int(parameters[0])
                    newlayers = int(voutput[1])
                    if oldlayers == newlayers:
                        return True
                        
                    if newlayers == 0:
                        start = 1
                        end = oldlayers
                        self.layersAboutToBeRemoved.emit(start, end)
                        
                        parameters.resize(8, refcheck = False)
                        thesignal = self.layersFinishedBeingRemoved
                        fitted_parameters = np.extract(fitted_parameters < 8, fitted_parameters)
                        parameters[0] = newlayers
                    else:
                        if newlayers > oldlayers:
                            title = 'Where would you like to insert the new layers'
                            maxValue = oldlayers
                            minValue = 0
                            value = 0
                        elif newlayers < oldlayers:
                            title = 'Where would you like to remove the layers from?'
                            maxValue = newlayers + 1
                            minValue = 1
                            value = 1
    
                        label = 'layer'                
                        insertpoint, ok = QtGui.QInputDialog.getInt(None,
                                                   title,
                                                    label,
                                                     value = value,
                                                      minValue = minValue,  
                                                       maxValue = maxValue)
                        if not ok:
                            return False
    
                        parameters[0] = newlayers
                        if newlayers > oldlayers:
                            start = insertpoint + 1
                            end = insertpoint + newlayers - oldlayers
                            self.layersAboutToBeInserted.emit(start, end)

                            parameters = np.insert(parameters,
                                                    [4 * insertpoint + 8] * 4 *(newlayers - oldlayers),
                                                     [0, 0, 0, 0] * (newlayers - oldlayers))
                            fitted_parameters = np.where(fitted_parameters >= 4 * insertpoint + 8,
                                                          fitted_parameters + (newlayers - oldlayers) * 4,
                                                             fitted_parameters)
                            fitted_parameters = np.append(fitted_parameters,
                                                     np.arange(4 * insertpoint + 8, 4 * insertpoint + 8 + (newlayers -oldlayers) * 4))

                            thesignal = self.layersFinishedBeingInserted
                        elif newlayers < oldlayers:
                            insertpoint -= 1
                            start = insertpoint + 1
                            end = insertpoint + 1 + (oldlayers - newlayers) - 1
                            self.layersAboutToBeRemoved.emit(start, end)
                                                       
                            paramslost = np.arange(4 * insertpoint + 8, 4 * insertpoint + 8 + (oldlayers - newlayers) * 4)
                            parameters = np.delete(parameters, paramslost)
                            fitted_parameters = np.array([val for val in fitted_parameters.tolist() if (val < paramslost[0] or val > paramslost[-1])])
                            fitted_parameters = np.where(fitted_parameters > paramslost[-1],
                                      fitted_parameters + (newlayers - oldlayers) * 4,
                                         fitted_parameters)
                            
                            thesignal = self.layersFinishedBeingRemoved

                            
                    #YOU HAVE TO RESIZE LAYER PARAMS
                    self.model.parameters = parameters[:]
                    self.model.fitted_parameters = fitted_parameters[:]
                    thesignal.emit()

                else:
                    return False
            else:
                validator = QtGui.QDoubleValidator()
                voutput = validator.validate(value, 1)
                if voutput[0] is QtGui.QValidator.State.Acceptable:
                    self.model.parameters[coltopar[index.column()]] = voutput[1]
                else:
                    return False
        
        self.dataChanged.emit(index, index)
        return True
                
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
            
        if not reflect.isProperAbelesInput(self.model.parameters):
            return None
        
        if index.row() != 0 or index.column() < 0 or index.column() > 2:
            return None
        
        coltopar = [0, 1, 6]
        
        if role == QtCore.Qt.DisplayRole:
            return str(self.model.parameters[coltopar[index.column()]])
        
        if role == QtCore.Qt.CheckStateRole:
            if coltopar[index.column()] in self.model.fitted_parameters and index.column() != 0:
                return QtCore.Qt.Unchecked
            else:
               return QtCore.Qt.Checked
                
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'layers'
            if section == 1:
                return 'scale factor'
            if section == 2:
                return 'background'
        return None
        
class LayerModel(QtCore.QAbstractTableModel):

    def __init__(self, model, parent = None):
        super(LayerModel, self).__init__(parent)
        self.model = model
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        return int(self.model.parameters[0]) + 2
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 4
    
    def flags(self, index):
        numlayers = int(self.model.parameters[0])
        row = index.row()
        col = index.column()
        
        if row == 0 and (col == 0 or col == 3):
        	return QtCore.Qt.NoItemFlags
        
        if row == numlayers + 1 and col == 0:
            return QtCore.Qt.NoItemFlags
            
    	return (QtCore.Qt.ItemIsEditable |
    	         QtCore.Qt.ItemIsUserCheckable |
    	           QtCore.Qt.ItemIsEnabled |
    	            QtCore.Qt.ItemIsSelectable)
    	            
    def rowcoltoparam(self, row, col, numlayers):
        if row == 0 and col == 1:
            param = 2
        elif row == 0 and col == 2:
            param = 3
        elif row == numlayers + 1 and col == 1:
            param = 4
        elif row == numlayers + 1 and col == 2:
            param = 5
        elif row == numlayers + 1 and col == 3:
            param = 7
        else:
            param = 4 * (row - 1) + col + 8
            
        return param
        
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
        numlayers = int(self.model.parameters[0])

        if not index.isValid():
            return False
                
        if col < 0 or col > 3:
            return False
                        
        param = self.rowcoltoparam(row, col, numlayers)
        
        if role == QtCore.Qt.CheckStateRole:
            fitted_parameters = self.model.fitted_parameters
            if value == QtCore.Qt.Checked:
                fitted_parameters = np.delete(fitted_parameters,np.where(fitted_parameters == param))
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
        
        if not reflect.isProperAbelesInput(self.model.parameters):
            return None

        numlayers = int(self.model.parameters[0])
                
        if row == 0 and (col == 0 or col == 3):
            return None       
        
        if row == numlayers + 1 and col == 0:
            return None
            
        param = self.rowcoltoparam(row, col, numlayers)

        if role == QtCore.Qt.DisplayRole:
            return str(self.model.parameters[param])
        
        if role == QtCore.Qt.CheckStateRole:
            if param in self.model.fitted_parameters:
                return QtCore.Qt.Unchecked
            else:
               return QtCore.Qt.Checked
                
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Vertical:
            numlayers = (self.model.parameters[0])
            if section == 0:
                return 'fronting'
            elif section == numlayers + 1:
                return 'backing'
            else:
                return str(section)
            
        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'thickness'
            if section == 1:
                return 'SLD / 1e-6'
            if section == 2:
                return 'iSLD'
            if section == 3:
                return 'roughness'
        return None
        
class LimitsModel(QtCore.QAbstractTableModel):
    def __init__(self, parameters, fitted_parameters, limits, parent = None):
        super(LimitsModel, self).__init__(parent)
        self.parameters = np.copy(parameters)
        self.fitted_parameters = np.unique(fitted_parameters)
        self.limits = np.copy(limits)
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 4
        
    def rowCount(self, parent = QtCore.QModelIndex()):
        return np.size(self.fitted_parameters)
        
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False
            
        row = index.row()
        col = index.column()
                    
        uniquevals = np.unique(self.fitted_parameters)
        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                return str(self.fitted_parameters[row])
            if col == 1:
                return str(self.parameters[self.fitted_parameters[row]])
            if col == 2:
                return str(self.limits[0, self.fitted_parameters[row]])
            if col == 3:
                return str(self.limits[1, self.fitted_parameters[row]])
                
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None
    
        if orientation == QtCore.Qt.Vertical:
            return ''            
        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'parameter'
            if section == 1:
                return 'value'
            if section == 2:
                return 'lower limit'
            if section == 3:
                return 'upper limit'
        return None

    def flags(self, index):
        row = index.row()
        col = index.column()
        
        if col == 0 or col == 1:
        	return QtCore.Qt.NoItemFlags
        
    	return (QtCore.Qt.ItemIsEditable |
    	           QtCore.Qt.ItemIsEnabled |
    	            QtCore.Qt.ItemIsSelectable)
    	            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()
    
        if not index.isValid():
            return False
                    
        if col < 0 or col > 3:
            return False
                                                
        if role == QtCore.Qt.EditRole:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(value, 1)
            if voutput[0] == QtGui.QValidator.State.Acceptable:
                if col == 2:
                    self.limits[0, self.fitted_parameters[row]] = voutput[1]
                if col == 3:
                    self.limits[1, self.fitted_parameters[row]] = voutput[1]                    
            else:
                return False
            
        self.dataChanged.emit(index, index)
        return True
        
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
                
    def add(self, filepath):
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
                    self.model.parameters[currentparams:] = 0.
                
                    if newparams > currentparams:
                        self.model.limits = np.append(self.model.limits, np.zeros((2, newparams - currentparams)),axis = 1)
                        defaultlimits = self.model.defaultlimits()
                        self.model.limits[:, currentparams:-1] = defaultlimits[:, currentparams:-1]
                        self.model.fitted_parameters = np.append(self.model.fitted_parameters, range(currentparams, newparams))
                        self.endInsertRows()
                    if newparams < currentparams:                
                        self.model.limits = self.model.limits[:, 0: newparams]
                        self.model.fitted_parameters.sort()
                        #get rid of all parameters greater than newparams
                        idx = np.searchsorted(self.model.fitted_parameters, newparams)
                        self.model.fitted_parameters = self.model.fitted_parameters[: idx]
                        self.endRemoveRows()   
                    self.modelReset.emit()                        
            if row > 0:
                validator = QtGui.QDoubleValidator()
                voutput = validator.validate(value, 1)
                if voutput[0] == QtGui.QValidator.State.Acceptable:
                    number = voutput[1]
                else:
                    print 'not true'
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
            limitssize = np.size(self.model.limits, 1)
            paramssize = np.size(self.model.parameters)
            currentlims = np.copy(self.model.limits)
            defaultlimits = self.model.defaultlimits()
            if limitssize != paramssize:
                if limitssize < paramssize:
                    self.model.limits = np.zeros((2, paramssize))
                    self.model.limits[:, 0: limitssize] = currentlims[:, 0:limitssize]
                elif limitssize > paramssize:
                    self.model.limits = np.zeros((2, paramssize))
                    self.model.limits[:, 0:paramssize] = currentlims[:, 0:paramssize]
                
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
                return str(section - 1)
            
        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'value'
            if section == 1:
                return 'lower limit'
            if section == 2:
                return 'upper limit'
        return None
                

