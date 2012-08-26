from __future__ import division
import reflectdataset
import numpy as np
import pyplatypus.analysis.reflect as reflect
from copy import deepcopy, copy
import matplotlib.artist as artist
from PySide import QtGui, QtCore

class DataStore(QtCore.QAbstractListModel):

    def __init__(self, parent = None):
        super(DataStore, self).__init__(parent)
        self.dataObjects = {}
        self.numDataObjects = 0
        self.names = []
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        #don't want to return '_theoretical_'
        return self.numDataObjects - 1
        
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return self.names[index.row()]
        
    def addDataObject(self, dataObject):
        self.dataObjects[dataObject.name] = dataObject
        if dataObject.name != '_theoretical_':
            self.names.append(dataObject.name)
        self.numDataObjects += 1
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
        
        self.fit = None
        self.residuals = None
        
        self.chi2 = -1
        self.sld_profile = None
        
        self.line2D = None
        self.line2D_properties = {}
        
        self.line2Dfit = None
        self.line2Dfit_properties = {}

        self.line2Dsld_profile = None
        self.line2Dsld_profile_properties = {}
        
    def __getstate__(self):
        self.__save_graph_properties()
        d = copy(self.__dict__)
        d['line2Dfit'] = None
        d['line2D'] = None
        d['line2Dsld_profile'] = None
        del(d['fit'])
        return d
        
        
    def __save_graph_properties(self):
        if self.line2D:
            for key in self.__requiredgraphproperties:
                self.line2D_properties[key] = artist.getp(self.line2D, key)

        if self.line2Dfit:
            for key in self.__requiredgraphproperties:
                self.line2Dfit_properties[key] = artist.getp(self.line2Dfit, key)
            
        if self.line2Dsld_profile:
            for key in self.__requiredgraphproperties:
                self.line2Dsld_profile_properties[key] = artist.getp(self.line2Dsld_profile, key)

        
    def do_a_fit(self, model):

        callerInfo = deepcopy(model.__dict__)
        callerInfo['xdata'] = self.W_q
        callerInfo['ydata'] = self.W_ref
        callerInfo['edata'] = self.W_refSD
        if thismodel.usedq:
            callerInfo['dqvals'] = self.W_qSD
        else:
            del(callerInfo['dqvals'])


        model.fitted_parameters = np.copy(model.fitted_parameters)        
        
        RFO = reflect.ReflectivityFitObject(**callerInfo)
        self.model.parameters, self.chi2 = RFO.fit()
        self.fit = RFO.model()
        self.residuals = self.fit - self.W_ref
        self.sld_profile = RFO.sld_profile()
        
                  
    def evaluate_chi2(self, model, store = False):
        
        callerInfo = deepcopy(model.__dict__)
        callerInfo['xdata'] = self.W_q
        callerInfo['ydata'] = self.W_ref
        callerInfo['edata'] = self.W_refSD
        if thismodel.usedq:
            callerInfo['dqvals'] = self.W_qSD
        else:
            del(callerInfo['dqvals'])

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
        if thismodel.usedq:
            callerInfo['dqvals'] = self.W_qSD  
        else:
            del(callerInfo['dqvals'])

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
                           
class Model(object):
    def __init__(self, parameters = None,
                    fitted_parameters = None,
                     limits = None,
                      useerrors = True,
                       usedq = True,
                        costfunction = reflect.costfunction_logR_noweight):
        self.parameters = parameters
        self.fitted_parameters = fitted_parameters
        self.useerrors = useerrors
        self.usedq = usedq
        self.limits = limits
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
        
    