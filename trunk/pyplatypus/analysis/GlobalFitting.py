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
            
    
    def __init__(self, fitObjectTuple, linkageArray, *args, **kwds):
        """
            FitObjectTuple is a tuple of fit objects.
            '''
                all we should have to do is override model for the FitObject class.
                
        """
        #def __init__(self, xdata, ydata, edata, fitfunction, parameters, *args, **kwds):
        
        totalydata = np.concatenate([fitObject.ydata for fitObject in fitObjectTuple])
        totaledata = np.concatenate([fitObject.edata for fitObject in fitObjectTuple])
        totalparams = np.concatenate([fitObject.parameters for fitObject in fitObjectTuple])
        self.FitObjectTuple = fitObjectTuple
        
        self.unique_pars, self.unique_pars_idx, self.unique_pars_inv
                                 = np.unique(linkageArray.astype('int32'),
                                    return_index = True,
                                        return_inverse = True)
                                 
        self.unique_pars_vector = totalparams[self.unique_pars_idx[self.unique_pars>=0]]
       
        if 'fitted_parameters' in kwds and kwds['fitted_parameters'] is not None:
            #if it's in kwds, then it'll get passed to the superclass constructor
            pass
        else:
            #initiate fitted_parameters from the individual fitObjects
            fitted_parameters = np.array([],dtype = 'int32')
            
            uniquelocs = self.unique_pars_idx[self.unique_pars>=0]
            
            for idx, fitObject in enumerate(fitObjectTuple):
                t1 = [(np.size(linkageArray, 1) * idx + x) in uniquelocs for x in fitObject.fitted_parameters]
                f = np.where(t1, linkageArray[idx, fitObject.fitted_parameters], -1)
                f = f[f>=0]
                fitted_parameters = np.r_[fitted_parameters, f]
            
            kwds['fitted_parameters'] = fitted_parameters                                            
         
        #initialise the FitObject superclass
        super(FitObject, self).__init__(None, totalydata, totaledata, None, unique_pars_vector, *args, **kwds)
       
                    
        if 'limits' in kwds and kwds['limits'] is not None and np.size(kwds['limits'], 1) == self.numparams:
            self.limits = kwds['limits']
        else:
            self.limits = np.zeros((2, self.numparams))
            self.limits[0, :] = 0
            self.limits[1, :] = 2 * self.parameters
        
        #limits for those that are varying.
        self.fitted_limits = self.limits[:, self.fitted_parameters]
        
        
    def model(self, parameters = None):
        '''
            calculate the model function for the global fit function
            params is a np.array that has the same size as self.parameters
        '''
        
        if parameters is not None:
            test_parameters = parameters
        else:
            test_parameters = self.parameters
        
        substituted_pars = test_parameters[self.unique_pars[self.unique_pars_inv]]
        
        off = lambda idx: idx * np.size(self.linkageArray, 1)
        
        evaluateddata = [x.model(params = substituted_pars[off(i) : off(i) + x.numparams)]
                             for i, x in enumerate(fitObjectTuple)]
        
        return np.r_[evaluateddata].flatten()