from __future__ import divide
from pyplatypus.reduce import reflectdataset
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
    
class dataObject(reflectdataset):
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
        
        self.is_visible = False
        self.symbol = None
