from __future__ import division
import pyplatypus.dataset.reflectdataset as reflectdataset
import numpy as np
import pyplatypus.analysis.reflect as reflect
import matplotlib.artist as artist
import os.path, os
import string
    
class Model(object):
    def __init__(self, parameters = None,
                    fitted_parameters = None,
                     limits = None,
                      useerrors = True,
                       usedq = True,
                        costfunction = reflect.costfunction_logR_noweight):
        self.parameters = np.copy(parameters)
        self.uncertainties = np.copy(parameters)
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
                    