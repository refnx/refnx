from __future__ import division
import numpy as np
import math
import scipy.linalg
from scipy.optimize import leastsq
from scipy.optimize import differential_evolution
import numdifftools as ndt


class FitAbortedException(Exception):

    '''
     An exception that the user can raise in their customised FitObjects
     to indicate that a fit has been aborted
    '''

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class FitObject(object):
    '''An object used to perform a curvefitting analysis.
    There are several ways of using this class.

    1) Instantiate the class 'as' is. Here you have to supply a fitfunction
    to calculate the theoretical model. Chi2 is minimised by default.
    However, you have the option of supplying a costfunction if you wish to
    minimise a different costmetric.

    2) Alternatively you can subclass the FitObject.
        option 1) Override the FitObject.model() method.
        If you override the FitObject.model() method, then you no longer
        have to supply a fitfunction.
        -OR-
        option 2) Override the FitObject.energy() method.
        If you override the FitObject.energy() method you no longer have to
        supply a fitfunction, or a costfunction. This method should specify
        how the cost metric varies as the fitted parameters vary.

    Note that if you use the LM option of this class's fit() method, then
    the energy() method is ignored. i.e. only options 1) and 2.1) are
    usable.
    '''

    def __init__(self, xdata, ydata, edata, fitfunction,
                 parameters, args=(), **kwds):
        '''
        Construction of the object initialises the data for the curve fit, but
        doesn't actually start it.

        xdata           - np.ndarray that contains the independent variables for
        the fit. This is not used by this class, other than pass it directly to
        the fitfunction.

        ydata[numpoints] - np.ndarray that contains the observations
        corresponding to each measurement point.

        edata[numpoints] - None or np.ndarray that contains the uncertainty
        (s.d.) for each of the observed y data points. (Use None if you do not
        have measured uncertainty on each point)

        fitfunction - callable function  of the form f(xdata, parameters,
        args=(), **kwds). The args tuple and kwds supplied in the construction
        of this FitObject are also passed in as extra arguments to the
        fitfunction. You can use None for fitfunction _IF_ you subclass this
        FitObject and provide your own energy method, or if you subclass the
        model method.

        parameters - np.ndarray that contains _all_ the parameters to be
        supplied to the fitfunction, not just those being fitted.

        args - this tuple can be used to pass extra arguments to
        FitObject.energy(), FitObject.model(), and the fitfunction and
        costfunction if they are not None.

        You may set the following optional parameters in kwds:

        fitted_parameters - an np.ndarray that contains the parameter numbers
        that are being fitted (the others are held)

        limits - an np.ndarray that contains the lower and upper limits for all
        the parameters. It should have shape (2, np.size(parameters)).

        costfunction - a callable costfunction with the signature
        costfunction(model, ydata, edata, parameters, *args). The fullset of
        parameters is passed, not just the ones being varied. Supply this
        function, or override the energy method of this class, to use something
        other than the default of chi2.

        Object attributes:
            self.xdata - see above for definition
            self.ydata - see above for definition
            self.edata - see above for definition
            self.fitfunction - see above for definition

            self.numpoints - the number of datapoints
            self.parameters - the entire set of parameters used for the fit
            (including those that vary). The fitting procedure overwrites this.
            self.numparams - total number of parameters
            self.costfunction - the costfunction to be used (optional)
            self.args - the args tuple supplied to the constructor
            self.kwds - the kwds dictionary supplied to the constructor
            self.fitted_parameters - the index of the parameters that are being
            allowed to vary

        :::NOTE:::
        The energy method is supplied by parameters that are being varied by the
        fit. i.e. something along the lines of
        self.parameters[self.fitted_parameters].
        This is a subset of the total number of parameters required to calculate
        the model. Therefore you need to do something like the following in the
        energy function (if you override it):

        #params are the values that are changing.
        test_parameters = np.copy(self.parameters)
        test_parameters[self.fitted_parameters] = params

        The model method is supplied by the entire set of parameters (those
        being held and those being varied).
        '''

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

        # need to set the seed for differential_evolution
        if 'seed' in kwds:
            self.seed = kwds['seed']

        if ('fitted_parameters' in kwds
             and kwds['fitted_parameters'] is not None):
            self.fitted_parameters = np.unique(
                np.copy(kwds['fitted_parameters']))
        else:
           # get rid of duplicate fitted parameters
            self.fitted_parameters = np.arange(self.numparams)

        if 'costfunction' in kwds:
            self.costfunction = kwds['costfunction']

        if ('limits' in kwds
            and kwds['limits'] is not None
            and np.size(kwds['limits'], 1) == self.numparams):

            self.limits = kwds['limits']
        else:
            self.limits = np.zeros((2, self.numparams))
            self.limits[0, :] = 0
            self.limits[1, :] = 2 * self.parameters

        # limits for those that are varying.
        self.fitted_limits = self.limits[:, self.fitted_parameters]

    def residuals(self, parameter_subset=None, *args):
        '''
        return the fit residuals for the fit object.
        '''
        test_parameters = np.copy(self.parameters)

        if not len(args):
            args = self.args

        if parameter_subset is not None:
            test_parameters[self.fitted_parameters] = parameter_subset

        modeldata = self.model(test_parameters, *args)

        sigma = np.atleast_1d(1.)
        if self.edata is not None:
            sigma = self.edata

        return (self.ydata - modeldata) / sigma

    def energy(self, parameter_subset=None, *args):
        '''
        The default cost function for the fit object is chi2 - the sum of
        the squared residuals divided by the error bars for each point.
        params - np.ndarray containing the parameters that are being fitted,
        i.e. this array is np.size(self.fitted_parameters) long.  If this is
        omitted the energy function uses the defaults that we supplied when
        the object was constructed.

        If you require a different cost function provide a subclass that
        overloads this method. An alternative is to provide the costfunction
        keyword to the constructor.

        Returns chi2 by default

        '''

        if not len(args):
            args = self.args

        if self.costfunction:
            test_parameters = np.copy(self.parameters)

            if parameter_subset is not None:
                test_parameters[self.fitted_parameters] = parameter_subset

            modeldata = self.model(test_parameters, *args)

            sigma = np.atleast_1d(1.)
            if self.edata is not None:
                sigma = self.edata

            return self.costfunction(modeldata,
                                     self.ydata,
                                     sigma,
                                     test_parameters,
                                     *args)
        else:
            residuals = self.residuals(parameter_subset, *args)
            return np.nansum(np.power(residuals, 2))

    def model(self, parameters, *args):
        '''
        calculate the theoretical model using the fitfunction.

        parameters - the full np.ndarray containing the parameters that are
        required for the fitfunction

        returns the theoretical model for the xdata, i.e.
        self.fitfunction(self.xdata, test_parameters, *args, **kwds)
        '''
        if not len(args):
            args = self.args

        return self.fitfunction(self.xdata, parameters, *args, **self.kwds)

    def fit(self, method=None):
        '''
        start the fit.  This method returns

        parameters, uncertainties, chi2 = FitObject.fit()

        method - select the fitting algorithm
                None = Differential Evolution
                'LM' = Levenberg Marquardt

        If you select 'LM', then normal least squares is performed and your cost
        function is ignored. In addition, the LM optimiser minimisers the sum of
        the square of the array returned by the residuals() method, it does not
        minimise the value returned by the energy() method. In practice this
        means you can't specify your own costfunction, or subclass energy().
        '''

        if method == 'LM':
            # do a Levenberg Marquardt fit instead

            initialparams = self.parameters[self.fitted_parameters]
            popt, pcov, infodict, mesg, ier = leastsq(self.residuals,
                                                      initialparams,
                                                      args=self.args,
                                                      maxfev=100000,
                                                      full_output=True)

            self.covariance = pcov
            self.parameters[self.fitted_parameters] = popt
            self.chi2 = np.sum(
                np.power(self.residuals(popt, *self.args), 2))
        else:
            bounds = zip(self.fitted_limits[0, :], self.fitted_limits[1, :])

            result = differential_evolution(self.energy,
                                            bounds,
                                            args=self.args,
                                            callback=self.callback,
                                            seed=self.seed)

            self.parameters[self.fitted_parameters] = result.x
            self.chi2 = result.fun

#             Hfun = ndt.Hessian(self.energy, n=2)
#             hess = Hfun(result.x)
#             self.covariance = scipy.linalg.pinv(hess)

#       if self.edata is None:
#             self.covariance *= self.chi2 / \
#                 (self.ydata.size - self.fitted_parameters.size)

        self.uncertainties = np.zeros(self.parameters.size)
#         self.uncertainties[self.fitted_parameters] = np.sqrt(
#             np.diag(self.covariance))

        return np.copy(self.parameters), np.copy(self.uncertainties), self.chi2

    def callback(self, xk, convergence = 0.):
        '''
        a default callback function for the fit object
        '''
        return True
