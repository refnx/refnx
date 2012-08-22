from __future__ import division
import reflectdataset
import numpy as np
import pyplatypus.analysis.reflect as reflect


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
            if key != '_theoretical_':
                self.dataObjects[key].refresh()
            
        
class dataObject(reflectdataset.ReflectDataset):        
    def __init__(self, dataTuple = None, name = '_theoretical_', fname = None, parameters = None, fitted_parameters = None):
        super(dataObject, self).__init__(dataTuple = dataTuple)

        self.name = '_theoretical_'
        
        if fname is not None:
            with open(fname, 'r') as f:
                self.load(f)
        
        self.fit = None
        self.residuals = None
        self.parameters = parameters
        self.fitted_parameters = fitted_parameters
        self.limits = None
        
        self.chi2 = -1
        self.sld_profile = None
        
        self.line2D = None
        self.line2Dfit = None
        self.line2Dsld_profile = None
    
    def do_a_fit(self, **kwds):
        theseparameters = self.parameters
        store = True

        keywords = {}
        keywords['costfunction'] = reflect.costfunction_logR_weight
        keywords['dqvals'] = self.W_qSD
        keywords['limits'] = None
        keywords['fitted_parameters'] = self.fitted_parameters
        
        if 'store' in kwds:
            store = kwds['store']
        if 'parameters' in kwds and kwds['parameters'] is not None:
            theseparameters = kwds['parameters']
        if 'fitted_parameters' in kwds:
            keywords['fitted_parameters'] = self.fitted_parameters = kwds['fitted_parameters']
        if 'limits' in kwds:
            keywords['limits'] = self.limits = kwds['limits']
        if 'dqvals' in kwds:
            keywords['dqvals'] = kwds['dqvals']
            
        RFO = reflect.ReflectivityFitObject(self.W_q, self.W_ref, self.W_refSD, theseparameters, **keywords)
        self.parameters, self.chi2 = RFO.fit()
        self.fit = RFO.model()
        self.residuals = self.fit - self.W_ref
        self.sld_profile = RFO.sld_profile()
        
                  
    def evaluate_chi2(self, **kwds):
        theseparameters = self.parameters
        store = False

        keywords = {}
        keywords['costfunction'] = reflect.costfunction_logR_weight
        keywords['dqvals'] = self.W_qSD
        
        if 'store' in kwds:
            store = kwds['store']
        if 'parameters' in kwds and kwds['parameters'] is not None:
            theseparameters = kwds['parameters']

        for key in kwds:
            if key in keywords:
                keywords[key] = kwds[key]
                
        RFO = reflect.ReflectivityFitObject(self.W_q, self.W_ref, self.W_refSD, theseparameters, **keywords)
        
        energy = RFO.energy() / self.numpoints
        if store:
            self.chi2 = energy
                
        return energy

    def evaluate_model(self, **kwds):   
        theseparameters = self.parameters
        costfunction = reflect.costfunction_logR_weight
        store = False     

        keywords = {}
        keywords['costfunction'] = reflect.costfunction_logR_weight
        keywords['dqvals'] = self.W_qSD
        
        if 'store' in kwds:
            store = kwds['store']
        if 'parameters' in kwds and kwds['parameters'] is not None:
            theseparameters = kwds['parameters']

        for key in kwds:
            if key in keywords:
                keywords[key] = kwds[key]

        RFO = reflect.ReflectivityFitObject(self.W_q,
                                             self.W_ref,
                                              self.W_refSD,
                                               theseparameters,
                                                **kwds)
                    
        model = RFO.model()
        sld_profile = RFO.sld_profile()
        if store:
            self.fit = model
            self.residuals = model - self.W_ref
            self.sld_profile = sld_profile

        return model, model - self.W_ref, sld_profile
            
