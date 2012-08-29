from __future__ import division
import reflectdataset
import numpy as np
import pyplatypus.analysis.reflect as reflect
from copy import deepcopy, copy
import matplotlib.artist as artist
from PySide import QtGui, QtCore

class DataStore(QtCore.QAbstractTableModel):

    def __init__(self, parent = None):
        super(DataStore, self).__init__(parent)
        self.dataObjects = {}
        self.numDataObjects = 0
        self.names = []
    
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
            return  QtCore.Qt.NoItemFlags
            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        if index.column() == 1:
            if value == QtCore.Qt.Checked:
                self.dataObjects[self.names[index.row()]].visible = True
            else:
                self.dataObjects[self.names[index.row()]].visible = False
                                
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
                if self.dataObjects[self.names[index.row()]].visible:
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
        TdataObject = dataObject()
        with open(filename, 'Ur') as f:
            TdataObject.load(f)
        
        #purge -ve values
        TdataObject.W_q = np.delete(TdataObject.W_q, np.where(TdataObject.W_ref < 0))
        TdataObject.W_refSD = np.delete(TdataObject.W_refSD, np.where(TdataObject.W_ref < 0))
        TdataObject.W_qSD = np.delete(TdataObject.W_qSD, np.where(TdataObject.W_ref < 0))
        TdataObject.W_ref = np.delete(TdataObject.W_ref, np.where(TdataObject.W_ref < 0))
        
        self.addDataObject(TdataObject)

        return TdataObject
    
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
            
        
class dataObject(reflectdataset.ReflectDataset):        
    __requiredgraphproperties = ['lw', 'label', 'linestyle', 'fillstyle', 'marker', 'markersize', 'markeredgecolor', 'markerfacecolor', 'zorder']
                                    
    def __init__(self, dataTuple = None, name = '_theoretical_', fname = None):
        super(dataObject, self).__init__(dataTuple = dataTuple)
        
        self.name = '_theoretical_'
        
        if fname is not None:
            with open(fname, 'Ur') as f:
                self.load(f)
        
        self.visible = True
        
        self.fit = None
        self.residuals = None
        
        self.chi2 = -1
        self.sld_profile = None
        
        self.line2D = None
        self.line2D_properties = {}
        
        self.line2Dfit = None
        self.line2Dfit_properties = {}

        self.line2Dresiduals = None
        self.line2Dresiduals_properties = {}
        
        self.line2Dsld_profile = None
        self.line2Dsld_profile_properties = {}
        
    def __getstate__(self):
        self.__save_graph_properties()
        d = copy(self.__dict__)
        d['line2Dfit'] = None
        d['line2D'] = None
        d['line2Dsld_profile'] = None
        d['line2Dresiduals'] = None
#        del(d['fit'])
        return d
        
        
    def __save_graph_properties(self):
        if self.line2D:
            for key in self.__requiredgraphproperties:
                self.line2D_properties[key] = artist.getp(self.line2D, key)

        if self.line2Dfit:
            for key in self.__requiredgraphproperties:
                self.line2Dfit_properties[key] = artist.getp(self.line2Dfit, key)

        if self.line2Dresiduals:
            for key in self.__requiredgraphproperties:
                self.line2Dresiduals_properties[key] = artist.getp(self.line2Dresiduals, key)
                            
        if self.line2Dsld_profile:
            for key in self.__requiredgraphproperties:
                self.line2Dsld_profile_properties[key] = artist.getp(self.line2Dsld_profile, key)

        
    def do_a_fit(self, model):

        callerInfo = deepcopy(model.__dict__)
        callerInfo['xdata'] = self.W_q
        callerInfo['ydata'] = self.W_ref
        callerInfo['edata'] = self.W_refSD
        
        try:
            if model.usedq:
                callerInfo['dqvals'] = self.W_qSD
            else:
                del(callerInfo['dqvals'])
        except KeyError:
            pass
                    
        RFO = reflect.ReflectivityFitObject(**callerInfo)
        model.parameters, self.chi2 = RFO.fit()
        self.fit = RFO.model()
        self.residuals = np.log10(self.fit/self.W_ref)
        self.sld_profile = RFO.sld_profile()
        
                  
    def evaluate_chi2(self, model, store = False):
        
        
        callerInfo = deepcopy(model.__dict__)
        callerInfo['xdata'] = self.W_q
        callerInfo['ydata'] = self.W_ref
        callerInfo['edata'] = self.W_refSD
        
        try:
            if model.usedq:
                callerInfo['dqvals'] = self.W_qSD
            else:
                del(callerInfo['dqvals'])
        except KeyError:
            pass
                    
        RFO = reflect.ReflectivityFitObject(**callerInfo)
        
        energy = RFO.energy() / self.numpoints
        if store:
            self.chi2 = energy
                
        return energy

    def evaluate_model(self, model, store = False):   
            
        callerInfo = deepcopy(model.__dict__)
        callerInfo['xdata'] = self.W_q
        callerInfo['ydata'] = self.W_ref
        callerInfo['edata'] = self.W_refSD
        
        try:
            if model.usedq:
                callerInfo['dqvals'] = self.W_qSD  
            else:
                del(callerInfo['dqvals'])
        except KeyError:
            pass
            
        RFO = reflect.ReflectivityFitObject(**callerInfo)
                               
        fit = RFO.model()
        sld_profile = RFO.sld_profile()
        if store:
            self.fit = fit
            self.residuals = fit - self.W_ref
            self.sld_profile = sld_profile

        return fit, fit - self.W_ref, sld_profile

class ModelStore(QtCore.QAbstractListModel):
    def __init__(self, parent = None):
        super(ModelStore, self).__init__(parent)
        self.models = {}
        self.names = []
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        #don't want to return '_theoretical_'
        return len(self.models.keys())
        
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return self.names[index.row()]
        
    def addModel(self, model, modelName):
        self.models[modelName] = model
        if modelName not in self.names:
            self.names.append(modelName)
        self.dataChanged.emit(QtCore.QModelIndex(),QtCore.QModelIndex())
                           
    def snapshot(self, name, snapshotname):
        model = self.models[name]
        snapshot = Model(parameters = model.parameters,
                            fitted_parameters = model.fitted_parameters,
                             limits = model.limits,
                              useerrors = model.useerrors,
                               costfunction = model.costfunction,
                                usedq = model.usedq)
        self.addModel(snapshot, snapshotname)

class Model(object):
    def __init__(self, parameters = None,
                    fitted_parameters = None,
                     limits = None,
                      useerrors = True,
                       usedq = True,
                        costfunction = reflect.costfunction_logR_noweight):
        self.parameters = np.copy(parameters)
        self.fitted_parameters = np.copy(fitted_parameters)
        self.useerrors = useerrors
        self.usedq = usedq
        self.limits = np.copy(limits)
        self.costfunction = costfunction
        
    def save(self, f):
        f.write(f.name + '\n\n')
        holdvector = np.ones_like(self.parameters)
        holdvector[self.fitted_parameters] = 0
        
        np.savetxt(f, np.column_stack((self.parameters, holdvector)))
    
    def load(self, f):
        h1 = f.readline()
        h2 = f.readline()
        array = np.loadtxt(f)
        self.parameters, a2 = np.hsplit(array, 2)
        self.parameters = self.parameters.flatten()
        a2 = a2.flatten()
        
        self.fitted_parameters = np.where(a2==0)[0]
        

class BaseModel(QtCore.QAbstractTableModel):

    layersInserted = QtCore.Signal(int, int)
    layersRemoved = QtCore.Signal(int, int)
    
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
                fitted_parameters = np.delete(fitted_parameters,np.where(fitted_parameters == col2par[index.column()]))
            else:
                fitted_parameters = np.append(fitted_parameters, col2par[index.column()])
                
            self.model.fitted_parameters = fitted_parameters[:]
            return True
            
        if role == QtCore.Qt.EditRole:
            if index.column() == 0:
                validator = QtGui.QIntValidator()
                voutput = validator.validate(value, 1)
                
                parameters = np.copy(self.model.parameters)
                fitted_parameters = np.copy(self.model.fitted_parameters)
                
                if voutput[0] is QtGui.QValidator.State.Acceptable and int(voutput[1]) >= 0:
                    oldlayers = int(parameters[0])
                    newlayers = int(voutput[1])
                    if oldlayers == newlayers:
                        return True
                        
                    if newlayers == 0:
                        parameters.resize(8, refcheck = False)
                        start = 1
                        end = oldlayers
                        thesignal = self.layersRemoved
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
                            parameters = np.insert(parameters,
                                                    [4 * insertpoint + 8] * 4 *(newlayers - oldlayers),
                                                     [0, 0, 0, 0] * (newlayers - oldlayers))
                            fitted_parameters = np.where(fitted_parameters >= 4 * insertpoint + 8,
                                                          fitted_parameters + (newlayers - oldlayers) * 4,
                                                             fitted_parameters)
                            fitted_parameters = np.append(fitted_parameters,
                                                     np.arange(4 * insertpoint + 8, 4 * insertpoint + 8 + (newlayers -oldlayers) * 4))
                            thesignal = self.layersInserted
                            start = insertpoint
                            end = insertpoint + newlayers - oldlayers - 1

                        elif newlayers < oldlayers:
                            insertpoint -= 1
                            
                            paramslost = np.arange(4 * insertpoint + 8, 4 * insertpoint + 8 + (oldlayers - newlayers) * 4)
                            parameters = np.delete(parameters, paramslost)
                            fitted_parameters = np.array([val for val in fitted_parameters.tolist() if (val < paramslost[0] or val > paramslost[-1])])
                            fitted_parameters = np.where(fitted_parameters > paramslost[-1],
                                      fitted_parameters + (newlayers - oldlayers) * 4,
                                         fitted_parameters)
                            thesignal = self.layersRemoved
                            start = insertpoint
                            end = insertpoint + (oldlayers - newlayers)
                            
                    #YOU HAVE TO RESIZE LAYER PARAMS
                    thesignal.emit(start, end)
                    self.model.parameters = parameters[:]
                    self.model.fitted_parameters = fitted_parameters[:]
                else:
                    self.errorHandler.showMessage("Number of layers must be integer > 0")
                    return False
            else:
                validator = QtGui.QDoubleValidator()
                voutput = validator.validate(value, 1)
                if voutput[0] is QtGui.QValidator.State.Acceptable:
                    self.model.parameters[coltopar(index.column())] = voutput[1]
                else:
                    print value
                    self.errorHandler.showMessage("values entered must be numeric")
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
        
    def layersInserted(self, start, end):
        self.beginInsertRows(QtCore.QModelIndex(), start, end)    
        self.endInsertRows()
        
    def layersRemoved(self, start, end):
        self.beginRemoveRows(QtCore.QModelIndex(), start, end)
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
                print value, row, col
                self.errorHandler.showMessage("values entered must be numeric")
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