from __future__ import division
import numpy as np
import math
import DEsolver
from scipy.optimize import leastsq 

def energy_for_fitting(params, *args):
    ''' 
        energy function for curve fitting.
        This energy function should work with DEsolver as well as the scipy.optimize modules.
        
        params are the parameters you are fitting.
        
        We have to pass in data, etc, through args. This args is passed through the optimize modules directly to 
        this energy function.
        
        The first argument in args should be a FitObject instance. This will have a method energy.

    '''
    
    return args[0].energy(params)
    

class FitObject(object):
    '''
        
        An object used to perform a curvefitting analysis.
        There are several ways of using this class.
        
        1) Instantiate the class 'as' is. Here you have to supply a fitfunction to calculate the theoretical model. Chi2 is minimised by default.
           However, you have the option of supplying a costfunction if you wish to minimise a different costmetric.
        
        2) Alternatively you can subclass the FitObject.
            option 1) Override the FitObject.model() method.
                    If you override the FitObject.model() method, then you no longer have to supply a fitfunction.
            -OR-
            option 2) Override the FitObject.energy() method.
                    If you override the FitObject.energy() method you no longer have to supply a fitfunction, or a costfunction. This method 
                    should specify how the cost metric varies as the fitted parameters vary.
            '''
            
    
    def __init__(self, xdata, ydata, edata, fitfunction, parameters, *args, **kwds):
        """
        
        Construction of the object initialises the data for the curve fit, but doesn't actually start it.
        
        xdata           - np.ndarray that contains the independent variables for the fit. This is not used by this class, other than pass it directly to the fitfunction.
                            
        ydata[numpoints] - np.ndarray that contains the observations corresponding to each measurement point.
        
        edata[numpoints] - np.ndarray that contains the uncertainty (s.d.) for each of the observed y data points.
        
        fitfunction - callable function  of the form f(xdata, parameters, *args, **kwds). The args and kwds supplied in the construction of this FitObject are also passed directly to the fitfunction and can be used to pass auxillary information to it. You can use None for fitfunction _IF_ you subclass this FitObject and provide your own energy method. Alternatively subclass the model method.
                        
        parameters - np.ndarray that contains _all_ the parameters to be supplied to the fitfunction, not just those being fitted.
        
        You may set the following optional parameters in kwds:
        
        fitted_parameters - an np.ndarray that contains the parameter numbers that are being fitted (the others are held)

        limits - an np.ndarray that contains the lower and upper limits for all the parameters. It should have shape (2, np.size(parameters)).
                
        costfunction - a callable costfunction with the signature costfunction(model, ydata, edata, parameters). The fullset of parameters is passed, not just the ones being varied. Supply this function, or override the energy method of this class, to use something other than the default of chi2.
        
        You can also choose to do a Levenberg Marquardt fit instead, by using the LMfit method.                 
                        
            Object attributes:
                self.xdata - see above for definition
                self.ydata - see above for definition
                self.edata - see above for definition
                self.fitfunction - see above for definition
                
                self.numpoints - the number of datapoints
                self.parameters - the entire set of parameters used for the fit (including those that vary). The fitting procedure overwrites this.
                self.numparams - total number of parameters
                self.costfunction - the costfunction to be used (optional)
                self.args - the args tuple supplied to the constructor
                self.kwds - the kwds dictionary supplied to the constructor
                self.fitted_parameters - the index of the parameters that are being allowed to vary 
                    :::NOTE:::
                        The energy method is supplied by parameters that are being varied by the fit. i.e. something along the lines of
                    self.parameters[self.fitted_parameters]. This is a subset of the total number of parameters required to calculate the model.
                    Therefore you need to do something like the following in the energy function (if you override it):
                        
                        #params are the values that are changing.
                        test_parameters = np.copy(self.parameters)
                        test_parameters[self.fitted_parameters] = params

                        The model method is supplied by the entire set of parameters (those being held and those being varied).

        
        """
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
        self.seed = None
        
        #need to set the seed for DEsolver.
        if 'seed' in kwds:
            self.seed = kwds['seed']       
        
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
            
    def energy(self, parameters = None):
        '''
            
            The default cost function for the fit object is chi2 - the sum of the squared residuals divided by the error bars for each point.
            params - np.ndarray containing the parameters that are being fitted, i.e. this array is np.size(self.fitted_parameters) long.
                    If this is omitted the energy function uses the defaults that we supplied when the object was constructed.
            
            If you require a diffe                  rent cost function provide a subclass that overloads this method. An alternative is to provide the costfunctioncfqaz
            keyword to the constructor.
            
            Returns chi2 by default
        
        '''
        test_parameters = np.copy(self.parameters)
    
        if parameters is not None:
            test_parameters[self.fitted_parameters] = parameters
        
        modeldata = self.model(test_parameters)
        
        if self.costfunction:
            return self.costfunction(modeldata, self.ydata, self.edata, test_parameters)
        else:
            #the following is required because the LMfit method requires only the residuals to be returned. Whereas the fit method utilising DE will require square energy.
            resid = (self.ydata - modeldata) / self.edata
            if self.square:
                return np.sum(np.power(resid, 2))
            else:
                return resid


    def model(self, parameters):
        '''
            
            calculate the theoretical model using the fitfunction.
            
            parameters - the full np.ndarray containing the parameters that are required for the fitfunction
            
            returns the theoretical model for the xdata, i.e. self.fitfunction(self.xdata, test_parameters, *args, **kwds)
            
        ''' 
                
        if parameters is not None:
            test_parameters = parameters
        else:
            test_parameters = self.parameters
        
        return self.fitfunction(self.xdata, test_parameters, *self.args, **self.kwds)
            

    def fit(self, method = None):
        '''
            start the fit.  This method returns 
            parameters, uncertainties, chi2 = FitObject.fit()            
            If method == 'LM', then a Levenberg Marquardt fit is used.
        '''
        self.square = True
        
        if method is None:
            de = DEsolver.DEsolver(energy_for_fitting,
                                     self.fitted_limits, (self),
                                         progress = self.progress,
                                          seed = self.seed)
            thefit, chi2 = de.solve()
            self.parameters[self.fitted_parameters] = thefit
            self.chi2 = chi2
            self.uncertainties = self.parameters + 0    

        elif method == 'LM':
            #do a Levenberg Marquardt fit instead
            self.square = False
            initialparams = self.parameters[self.fitted_parameters]
            stuff = leastsq(energy_for_fitting,
                                  initialparams,
                                   args = (self),
                                    full_output = True) 
            popt = stuff[0]
            pcov = stuff[1]
            self.covariance = pcov
            self.parameters[self.fitted_parameters] = popt
            self.chi2 = np.sum(np.power(self.energy(), 2))
            self.uncertainties = np.zeros(self.parameters.size)
            self.uncertainties[self.fitted_parameters] = np.diag(pcov)
            
        return np.copy(self.parameters), np.copy(self.uncertainties), self.chi2
 
        
    def progress(self, iterations, convergence, chi2, *args):
        '''
            a default progress function for the fit object
        '''
        return True
        