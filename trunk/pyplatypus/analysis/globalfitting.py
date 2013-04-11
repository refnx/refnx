from __future__ import division
import numpy as np
import math
import DEsolver
import fitting
import reflect

class GlobalFitObject(fitting.FitObject):
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
        
        self.linkageArray = np.atleast_2d(linkageArray)
        
        totalydata = np.concatenate([fitObject.ydata for fitObject in fitObjectTuple])
        totaledata = np.concatenate([fitObject.edata for fitObject in fitObjectTuple])
        totalparams = np.concatenate([fitObject.parameters for fitObject in fitObjectTuple])
        self.FitObjectTuple = fitObjectTuple
        
        self.unique_pars, self.unique_pars_idx, self.unique_pars_inv = np.unique(self.linkageArray.astype('int32'),
                                    return_index = True,
                                        return_inverse = True)
                                 
        self.unique_pars_vector = totalparams[self.unique_pars_idx[self.unique_pars>=0]]
        uniquelocs = self.unique_pars_idx[self.unique_pars>=0]
       
        '''
            sort out which parameters are to be fitted.
            If you supply an np.array, fitted_parameters in kwds, then the code will use 
            that. But, it will have to make sense compared to unique_pars_vector (i.e. no
            fitted parameter number > unique_pars_vector.size -1.
            Alternatively it will fit the parameters listed in the individual fitObject.fitted_parameters
            arrays IFF they are unique parameters.  Note that when you set up the individual fitObject
            if you don't supply the fitted_parameters keyword, then the default is to fit them all.
        '''
        
        if 'fitted_parameters' in kwds and kwds['fitted_parameters'] is not None:
            #if it's in kwds, then it'll get passed to the superclass constructor
            pass
        else:
            #initiate fitted_parameters from the individual fitObjects
            fitted_parameters = np.array([],dtype = 'int32')
            
            for idx, pos in enumerate(uniquelocs):
                row = int(pos // np.size(self.linkageArray, 1))
                col = pos%(np.size(self.linkageArray, 1))
                if col in fitObjectTuple[row].fitted_parameters:
                    fitted_parameters = np.append(fitted_parameters, idx)
            
            kwds['fitted_parameters'] = fitted_parameters                                            
        
        '''
        If you supply the limits array in kwds, then the code will use that. But it has to
         make sense with respect to the size of self.unique_pars_vector:
         The shape of limits should be limits.shape = (2, N)
         The shape of unique_pars_vector should be unique_pars_vector.shape = N
         In other words, each parameter has an upper and lower value.
         If the limits array is not supplied, then each parameter in unique_pars_vector will 
         use the limits from the individual fitObject that it came from. When you setup the
         individual fitObject if you don't supply the limits keyword, then the default is
         0 and 2 times the initial parametervalue.
        '''
        if 'limits' in kwds and kwds['limits'] is not None and np.size(kwds['limits'], 1) == self.unique_pars_vector.size:
            #self.limits gets setup in the superclass constructor
            pass
        else:
            #setup limits from individual fitObject
            limits = np.zeros((2, self.unique_pars_vector.size))
            
            for idx, pos in enumerate(uniquelocs):
                row = int(pos // np.size(self.linkageArray, 1))
                col = pos%(np.size(self.linkageArray, 1))
                limits[0, idx] = fitObjectTuple[row].limits[0, col]
                limits[1, idx] = fitObjectTuple[row].limits[1, col]
            
            kwds['limits'] = limits
                     
        #initialise the FitObject superclass
        super(GlobalFitObject, self).__init__(None, totalydata, totaledata, None, self.unique_pars_vector, *args, **kwds)
        self.fitObjectTuple = fitObjectTuple
                                    
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
        
        evaluateddata = [x.model(parameters = substituted_pars[off(i) : off(i) + x.numparams]) for i, x in enumerate(self.fitObjectTuple)]
        
        return np.r_[evaluateddata].flatten()
        
def test_globfit_modelvals_degenerate_layers():
    '''
        try fitting dataset with two degenerate layers
    '''
    np.seterr(invalid='raise')
    theoretical = np.loadtxt('pyplatypus/analysis/test/c_PLP0011859_q.txt')

    qvals, rvals, evals, dummy = np.hsplit(theoretical, 4)

    coefs = np.zeros((20))
    coefs[0] = 3
    coefs[1] = 1.
    coefs[2] = 2.07
    coefs[4] = 6.36
    coefs[6] = 2e-6
    coefs[7] = 3
    coefs[8] = 30
    coefs[9] = 3.47
    coefs[11] = 4
    coefs[12] = 125
    coefs[13] = 2
    coefs[15] = 4
    coefs[16] = 125
    coefs[17] = 2
    coefs[19] = 4

    
    fitted_parameters = np.array([3,5,6,7,8,9,10,11,12,13,14,15,16,17,19])
    
    a = reflect.ReflectivityFitObject(qvals, rvals, evals, coefs, fitted_parameters = fitted_parameters)
    linkageArray = np.arange(20)
    linkageArray[16] = 12
    linkageArray[17] = 16
    linkageArray[18] = 17
    linkageArray[19] = 18
    
    gfo = GlobalFitObject(tuple([a]), linkageArray)
    pars, dummy, chi2 = gfo.fit() 


if __name__ == '__main__':
    test_globfit_modelvals_degenerate_layers()