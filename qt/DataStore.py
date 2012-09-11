from __future__ import division
import pyplatypus.dataset.reflectdataset as reflectdataset
import numpy as np
import pyplatypus.analysis.reflect as reflect
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
        
    def saveDataStore(self, folderName):
        
        for key in self.dataObjects.keys():   
            dataObject = self.dataObjects[key]
            try:
                filename = os.path.join(folderName, dataObject.name)
            except AttributeError:
                print folderName, key
                
            with open(filename, 'w') as f:
                dataObject.save(f)
                
    def loadDataStore(self, folderName, clear = False):
        if clear:
            self.dataObjects.clear()
            self.names = []
            self.numDataObjects = 0
            
        filelist = os.listdir(folderName)
        for filename in filelist:
            try:
                self.loadDataObject(os.path.join(folderName, filename))
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
        self._save_graph_properties()
        d = copy(self.__dict__)
        d['line2Dfit'] = None
        d['line2D'] = None
        d['line2Dsld_profile'] = None
        d['line2Dresiduals'] = None
#        del(d['fit'])
        return d
        
    def save(self, f):
        #this will save it as XML
        super(dataObject, self).save(f)
        if self.fit is None:
            return
            
        #have to add in extra bits about the fit.
        try:
            tree = ET.ElementTree()    
            tree.parse(f)
        except Exception as inst:
            print type(inst)
        
        try:  
            refdata = tree.find('.//REFdata')
            fit = ET.SubElement(refdata, 'fit')
            fit.text = string.translate(repr(self.fit.tolist()), None, ',[]')
            tree.write(f)
        except Exception as inst:
            print type(inst)
                                   
    def _save_graph_properties(self):
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
        
        self.progressdialog = QtGui.QProgressDialog("Fit progress", "Abort", 0, 100)   
        self.progressdialog.setWindowModality(QtCore.Qt.WindowModal)

        RFO = reflect.ReflectivityFitObject(**callerInfo)
        RFO.progress = self.progress
        model.parameters, self.chi2 = RFO.fit()
        
        self.progressdialog.setValue(100)
        
        self.fit = RFO.model()
        self.residuals = np.log10(self.fit/self.W_ref)
        self.sld_profile = RFO.sld_profile()
    
    def progress(self, iterations, convergence, chi2, *args):
        self.progressdialog.setValue(int(convergence * 100))
        if self.progressdialog.wasCanceled():
            return False
        else:  
            return True
                  
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
        
    def saveModelStore(self, folderName):
        for modelname in self.names:
            model = self.models[modelname]
            filename = os.path.join(folderName, modelname)
            with open(filename, 'w+') as f:
                model.save(f)
                
    def loadModelStore(self, folderName, clear = False):
        if clear:
            self.models.clear()
            self.names = []
            
        filelist = os.listdir(folderName)
        for filename in filelist:
            try:
                with open(os.path.join(folderName, filename), 'Ur') as f:
                    model = Model()
                    model.load(f)
                    self.addModel(model, filename)
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
        
        if self.limits is None or self.limits.ndim != 2 or np.size(self.limits, 1) != np.size(self.parameters):
            self.defaultlimits()
            
        np.savetxt(f, np.column_stack((self.parameters, holdvector, self.limits.T)))
    
    def load(self, f):
        h1 = f.readline()
        h2 = f.readline()
        array = np.loadtxt(f)
        self.parameters, a2, lowlim, hilim = np.hsplit(array, 4)
        self.parameters = self.parameters.flatten()
        self.limits = np.column_stack((lowlim, hilim)).T
        
        a2 = a2.flatten()
        
        self.fitted_parameters = np.where(a2==0)[0]
        
    def defaultlimits(self):
        self.limits = np.zeros((2, np.size(self.parameters)))
            
        for idx, val in enumerate(self.parameters):
            if val < 0:
                self.limits[0, idx] = 2 * val
            else:
                self.limits[1, idx] = 2 * val 
                    
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