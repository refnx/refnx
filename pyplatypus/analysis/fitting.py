from __future__ import division
import numpy as np
import math
import DEsolver
import scipy.linalg
from scipy.optimize import leastsq 
import numdifftools as ndt

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
                    If you override the FitObject.energy() method you no longer have to supply a fitfunction, or a costfunction. This method should specify how the cost metric varies as the fitted parameters vary.
            '''
            
    
    def __init__(self, xdata, ydata, edata, fitfunction, parameters, args = (), **kwds):
        """
        
        Construction of the object initialises the data for the curve fit, but doesn't actually start it.
        
        xdata           - np.ndarray that contains the independent variables for the fit. This is not used by this class, other than pass it directly to the fitfunction.
                            
        ydata[numpoints] - np.ndarray that contains the observations corresponding to each measurement point.
        
        edata[numpoints] - None or np.ndarray that contains the uncertainty (s.d.) for each of the observed y data points. (Use None if you do not have measured uncertainty on each point)
        
        fitfunction - callable function  of the form f(xdata, parameters, args = (), **kwds). The args tuple and kwds supplied in the construction of this FitObject are also passed in as extra arguments to the fitfunction. You can use None for fitfunction _IF_ you subclass this FitObject and provide your own energy method, or if you subclass the model method.
                        
        parameters - np.ndarray that contains _all_ the parameters to be supplied to the fitfunction, not just those being fitted.
        
        args - this tuple can be used to pass extra arguments to FitObject.energy(), FitObject.model(), and the fitfunction and costfunction if they are not None.
        
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
        if edata is not None:
            self.edata = np.copy(edata.flatten())
        else:
            self.edata = None

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
        
    def residuals(self, parameters = None):
        test_parameters = np.copy(self.parameters)

        if parameters is not None:
            test_parameters[self.fitted_parameters] = parameters

        modeldata = self.model(test_parameters, args = self.args)

        if self.edata is None:
            sigma = np.atleast_1d(1.)
        else:
            sigma = self.edata

        return (self.ydata - modeldata) / sigma

            
    def energy(self, parameters = None, args = ()):
        '''
            
            The default cost function for the fit object is chi2 - the sum of the squared residuals divided by the error bars for each point.
            params - np.ndarray containing the parameters that are being fitted, i.e. this array is np.size(self.fitted_parameters) long.
                    If this is omitted the energy function uses the defaults that we supplied when the object was constructed.
            
            If you require a different cost function provide a subclass that overloads this method. An alternative is to provide the costfunction keyword to the constructor.
            
            Note that fitting with Levenberg Marquardt assumes that this function returns a vector containing residuals. The fit method sets an object attribute self._LMleastsquare=True if Levenberg Marquardt fitting is selected.
            
            Returns chi2 by default
            
        
        '''
        if not len(args):
            args = self.args
            
        test_parameters = np.copy(self.parameters)
    
        if parameters is not None:
            test_parameters[self.fitted_parameters] = parameters
        
        modeldata = self.model(test_parameters, *args)
        
        if self.edata is None:
            sigma = np.atleast_1d(1.)
        else:
            sigma = self.edata
        
        if self.costfunction:
            return self.costfunction(modeldata, self.ydata, sigma, test_parameters, *args)
        else:
            #the following is required because the LMfit method requires only the residuals to be returned. Whereas the fit method utilising DE will require square energy.
            resid = (self.ydata - modeldata) / sigma
            if not self._LMleastsquare:
                return np.sum(np.power(resid, 2))
            else:
                return resid                

    def model(self, parameters, args = ()):
        '''
            
            calculate the theoretical model using the fitfunction.
            
            parameters - the full np.ndarray containing the parameters that are required for the fitfunction
            
            returns the theoretical model for the xdata, i.e. self.fitfunction(self.xdata, test_parameters, *args, **kwds)
            
        ''' 
        if not len(args):
            args = self.args
        
        if parameters is not None:
            test_parameters = parameters
        else:
            test_parameters = self.parameters
        
        return self.fitfunction(self.xdata, test_parameters, *args, **self.kwds)
            

    def fit(self, method = None):
        '''
            start the fit.  This method returns 
            parameters, uncertainties, chi2 = FitObject.fit()            
            If method == 'LM', then a Levenberg Marquardt fit is used.
        '''
        self._LMleastsquare = False
        
        if method is None:
            de = DEsolver.DEsolver(self.energy,
                                     self.fitted_limits, args = self.args,
                                         progress = self.progress,
                                          seed = self.seed)
            popt, chi2 = de.solve()
            self.parameters[self.fitted_parameters] = popt
            self.chi2 = chi2
            Hfun = ndt.Hessian(self.energy, n=2)
            hess = Hfun(popt)
            self.covariance = scipy.linalg.pinv(hess)

        elif method == 'LM':
            #do a Levenberg Marquardt fit instead
            self._LMleastsquare = True
            initialparams = self.parameters[self.fitted_parameters]
            stuff = leastsq(self.energy,
                                  initialparams,
                                   args = self.args,
                                   maxfev=100000,
                                    full_output = True) 
            popt = stuff[0]
            pcov = stuff[1]
            
            self.covariance = pcov
            self.parameters[self.fitted_parameters] = popt
            self.chi2 = np.sum(np.power(self.energy(), 2))

        if self.edata is None:
            self.covariance *= self.chi2 / (self.ydata.size - self.fitted_parameters.size)

        self.uncertainties = np.zeros(self.parameters.size)
        self.uncertainties[self.fitted_parameters] = np.sqrt(np.diag(self.covariance))
                
        return np.copy(self.parameters), np.copy(self.uncertainties), self.chi2
 
        
    def progress(self, iterations, convergence, chi2, args = ()):
        '''
            a default progress function for the fit object
        '''
        return True
        