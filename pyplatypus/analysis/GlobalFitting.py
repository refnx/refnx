from __future__ import division
import numpy as np
import math
import DEsolver
import fitting
import reflect

class GlobalFitObject(FitObject):
    '''
        
        An object used to perform a global curvefitting analysis.
            '''
            
    
    def __init__(self, fitObjectTuples, linkageMatrix, *args, **kwds):
        """
            FitObjectTuples is a tuple of fit objects.
            '''
                all we should have to do is override model for the FitObject class.
                
        """
        #def __init__(self, xdata, ydata, edata, fitfunction, parameters, *args, **kwds):
        
        
        super(FitObject, self).__init__(None, None, None, None, None)
        self.FitObjectTuples = fitObjectTuples
        self.xdata = np.copy(xdata)
        self.ydata = np.copy(ydata.flatten())
        self.edata = np.copy(edata.flatten())
        self.numpoints = np.size(ydata, 0)
        
        self.fitfunction = fitfunction
        self.parameters = np.copy(parameters)
        self.numparams = np.size(parameters, 0)
        self.costfunction = None
        self.args = args
        self.kwds = kwds
        
        if 'fitted_parameters' in kwds and kwds['fitted_parameters'] is not None:
            self.fitted_parameters = np.copy(kwds['fitted_parameters'])
        else:
            self.fitted_parameters = np.arange(self.numparams)
            
        if 'costfunction' in kwds:
            self.costfunction = kwds['costfunction']
        
        if 'limits' in kwds and kwds['limits'] is not None and np.size(kwds['limits'], 1) == self.numparams:
            self.limits = kwds['limits']
        else:
            self.limits = np.zeros((2, self.numparams))
            self.limits[0, :] = 0
            self.limits[1, :] = 2 * self.parameters
        
        #limits for those that are varying.
        self.fitted_limits = self.limits[:, self.fitted_parameters]
        
        
    def model(self):
        pass
