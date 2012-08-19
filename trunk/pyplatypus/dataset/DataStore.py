from __future__ import division
import reflectdataset
import numpy as np


class DataStore(object):

    def __init__(self):
        self.dataObjects = {}
        self.numDataObjects = 0
        return
        
    def addDataObject(self, dataObject):
        self.dataObjects[dataObject.name] = dataObject
        self.numDataObjects += 1
        
    def loadDataObject(self, filename):
        TdataObject = dataObject()
        with open(filename, 'r') as f:
            TdataObject.load(f)
            
        self.addDataObject(TdataObject)
        return TdataObject
                      
    def getDataObject(self, name):
        return self.dataObjects[name]
        
    def removeDataObject(self, name):
        del(self.dataObjects[name])
        
    def refresh(self):
        for key in self.dataObjects:
            self.dataObjects[key].refresh()
            
        
    
class dataObject(reflectdataset.ReflectDataset):        
    def __init__(self, fname = None):
        super(dataObject, self).__init__()

        self.fit = None
        self.residuals = None
        self.model = None
        self.holdvector = None
        self.limits = None
        self.chi2 = -1
        self.sld_profile = None
        
        self.line2D = None


