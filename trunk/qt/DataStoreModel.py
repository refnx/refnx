from __future__ import division
import pyplatypus.dataset.reflectdataset as reflectdataset
import numpy as np
import pyplatypus.analysis.reflect as reflect
import pyplatypus.dataset.DataObject as DataObject
import pyplatypus.analysis.Model as Model
from copy import deepcopy, copy
import matplotlib.artist as artist
from PySide import QtGui, QtCore
import os.path, os
import string
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def zipper(dir, zip):
    root_len = len(os.path.abspath(dir))
    for root, dirs, files in os.walk(dir):
        archive_root = os.path.abspath(root)[root_len:]
        for f in files:
            fullpath = os.path.join(root, f)
            archive_name = os.path.join(archive_root, f)
            zip.write(fullpath, archive_name)
    
class DataStore(QtCore.QAbstractTableModel):

    def __init__(self, parent = None):
        super(DataStore, self).__init__(parent)
        self.dataObjects = {}
        self.numDataObjects = 0
        self.names = []
        
    def __iter__(self):
        dataObjects = [self.dataObjects[name] for name in self.names]
        for dataObject in dataObjects:
            yield dataObject
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        #don't want to return '_theoretical_'
        return self.numDataObjects - 1
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 2
    
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
                self.dataObjects[self.names[index.row()]].graph_properties['visible'] = True
            else:
                self.dataObjects[self.names[index.row()]].graph_properties['visible'] = False
                                
            self.dataChanged.emit(index, index)
        
        return True
                
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        
        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0:
                return self.names[index.row()]
        
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
            
    def addDataObject(self, dataObject):
        self.dataObjects[dataObject.name] = dataObject
        if dataObject.name != '_theoretical_':
            self.names.append(dataObject.name)
        self.numDataObjects += 1
        self.insertRows(self.numDataObjects - 1)
        
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())
                
    def loadDataObject(self, filename):
        if os.path.basename(filename) in self.names:
            self.dataObjects[os.path.basename(filename)].refresh()
            return None
            
        TdataObject = DataObject.DataObject()
        with open(filename, 'Ur') as f:
            TdataObject.load(f)
        
        #purge -ve values
        TdataObject.W_q = np.delete(TdataObject.W_q, np.where(TdataObject.W_ref < 0))
        TdataObject.W_refSD = np.delete(TdataObject.W_refSD, np.where(TdataObject.W_ref < 0))
        TdataObject.W_qSD = np.delete(TdataObject.W_qSD, np.where(TdataObject.W_ref < 0))
        TdataObject.W_ref = np.delete(TdataObject.W_ref, np.where(TdataObject.W_ref < 0))
        
        self.addDataObject(TdataObject)

        return TdataObject
        
    def saveDataStore(self, folderName):
        
        for key in self.dataObjects.keys():   
            dataObject = self.dataObjects[key]
            try:
                filename = os.path.join(folderName, dataObject.name)
            except AttributeError:
                print folderName, key
            
            with open(filename, 'w') as f:
                pass
                
            with open(filename, 'r+') as f:
                dataObject.save(f)
                
    def loadDataStore(self, files, clear = False):
        if clear:
            self.dataObjects.clear()
            self.names = []
            self.numDataObjects = 0
            
        for file in files:
            try:
                self.loadDataObject(file)
            except IOError:
                continue
        
        self.modelReset.emit()       
                       
    def snapshot(self, name, snapshotname):
        #this function copies the data from one dataobject into another.
        origin = self.getDataObject(name)
        dataTuple = (np.copy(origin.W_q), np.copy(origin.W_ref), np.copy(origin.W_refSD), np.copy(origin.W_qSD))
        snapshot = dataObject(name = snapshotname, dataTuple = dataTuple)
        self.addDataObject(snapshot)
                         
    def getDataObject(self, name):
        return self.dataObjects[name]
        
    def removeDataObject(self, name):
        del(self.dataObjects[name])
        
    def refresh(self):
        for key in self.dataObjects:
            if key != '_theoretical_':
                self.dataObjects[key].refresh()
                 
class ModelStore(QtCore.QAbstractListModel):
    def __init__(self, parent = None):
        super(ModelStore, self).__init__(parent)
        self.models = {}
        self.names = []
        self.displayOtherThanReflect = False
    
    def __iter__(self):
        models = [self.models[name] for name in self.names]
        for model in models:
            yield model
                    
    def rowCount(self, parent = QtCore.QModelIndex()):
        #don't want to return '_theoretical_'
        return len(self.models.keys())
        
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return self.names[index.row()]
            
    def flags(self, index, filterNormalRef = True):
        parameters = self.models[self.names[index.row()]].parameters
        if self.displayOtherThanReflect:
        	return (QtCore.Qt.ItemIsEnabled |
    	            QtCore.Qt.ItemIsSelectable)
        else:
            if reflect.isProperAbelesInput(parameters):
            	return (QtCore.Qt.ItemIsEnabled |
        	            QtCore.Qt.ItemIsSelectable)
            else:
            	return (QtCore.Qt.NoItemFlags)

        
    def addModel(self, model, modelName):
        self.models[modelName] = model
        if modelName not in self.names:
            self.names.append(modelName)
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())
        
    def saveModelStore(self, folderName):
        for modelname in self.names:
            model = self.models[modelname]
            filename = os.path.join(folderName, modelname)
            with open(filename, 'w+') as f:
                model.save(f)
                
    def loadModelStore(self, files, clear = False):
        if clear:
            self.models.clear()
            self.names = []
            
        for file in files:
            try:
                with open(file, 'Ur') as f:
                    model = Model()
                    model.load(f)
                    self.addModel(model, os.path.basename(file))
            except IOError:
                #may be a directory
                continue
        
        self.modelReset.emit()       
 
    def snapshot(self, name, snapshotname):
        model = self.models[name]
        snapshot = Model(parameters = model.parameters,
                            fitted_parameters = model.fitted_parameters,
                             limits = model.limits,
                              useerrors = model.useerrors,
                               costfunction = model.costfunction,
                                usedq = model.usedq)
        self.addModel(snapshot, snapshotname)

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
                    raise InputError("The size of the parameter array passed to abeles should be 4 * coefs[0] + 8")
                
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
        