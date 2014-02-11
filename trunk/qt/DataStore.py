from __future__ import division
import numpy as np
import DataObject
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

class DataStore(object):
    def __init__(self):
        super(DataStore, self).__init__()
        self.dataObjects = {}
        self.numDataObjects = 0
        self.names = []
        
    def __getitem__(self, key):
        if key in self.names:
            return self.dataObjects[key]
        return None
        
    def __iter__(self):    
        for key in self.dataObjects:
            yield self.dataObjects[key]
        
    def add(self, dataObject):
        self.dataObjects[dataObject.name] = dataObject
        self.names.append(dataObject.name)
        self.numDataObjects += 1

    def load(self, filename):
        if os.path.basename(filename) in self.names:
            self.dataObjects[os.path.basename(filename)].refresh()
            return None
            
        TdataObject = DataObject.DataObject()
        with open(filename, 'Ur') as f:
            TdataObject.load(f)
                
        self.add(TdataObject)

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
                self.loadData(file)
            except IOError:
                continue
                         
    def snapshot(self, name, snapshotname):
        #this function copies the data from one dataobject into another.
        origin = self.getDataObject(name)
        dataTuple = (np.copy(origin.W_q), np.copy(origin.W_ref), np.copy(origin.W_refSD), np.copy(origin.W_qSD))
        snapshot = dataObject(name = snapshotname, dataTuple = dataTuple)
        self.addDataObject(snapshot)
        
    def removeDataObject(self, name):
        del(self.dataObjects[name])
        del(self.names[self.names.index(name)])
        self.numDataObjects -= 1
        
    def refresh(self):
        for dataObject in self:
            dataObject.refresh()


class ModelStore(object):
    def __init__(self):
        super(ModelStore, self).__init__()
        self.models = {}
        self.names = []
        self.displayOtherThanReflect = False
        
    def __getitem__(self, key):
        return self.models[key]
    
    def __iter__(self):
        for key in self.models:
            yield self.models[key]
    
    def add(self, model, modelName):
        self.models[modelName] = model
        if modelName not in self.names:
            self.names.append(modelName)
            
    def saveModelStore(self, folderName):
        for modelname in self.names:
            model = self.models[modelname]
            filename = os.path.join(folderName, modelname)
            with open(filename, 'wb+') as f:
                model.save(f)
                
    def loadModelStore(self, files, clear = False):
        if clear:
            self.models.clear()
            self.names = []
            
        for file in files:
            try:
                with open(file, 'Ur') as f:
                    model = Model.Model(None)
                    model.load(f)
                    self.add(model, os.path.basename(file))
            except IOError:
                #may be a directory
                continue  
                
    def snapshot(self, name, snapshotname):
        model = self.models[name]
        snapshot = Model(parameters = model.parameters,
                            fitted_parameters = model.fitted_parameters,
                             limits = model.limits,
                              useerrors = model.useerrors,
                               costfunction = model.costfunction,
                                usedq = model.usedq)
        self.add(snapshot, snapshotname)

                                                  
