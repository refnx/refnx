from __future__ import divide
from import pyplatypus.dataset.data_1D import Data_1D
import numpy as np

class DataStore(object):

    def __init__(self):
        self.dataObjects = {}
        self.numDataObjects = 0
        return
        
    def addDataObject(self, dataObject):
        self.dataObjects[dataObject.name] = dataObject
        self.numDataObjects += 1
        pass
        
    def getDataObject(self, name):
        return self.dataObjects[name]
        
    def removeDataObject(self, name):
        del(self.dataObjects[name])
    
class dataObject(Data_1D):
    def __init__(self, fname):
        super(dataObject, self).__init__()
        self.read_dat(fname)

        self.fit = None
        self.residuals = None
        self.model = None
        self.holdvector = None
        self.limits = None
        self.chi2 = -1
        self.sld_profile = None
        
        self.isDisplayed = True
        self.symbol = None
