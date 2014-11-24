from __future__ import division
import numpy as np
import math
import scipy
from scipy.optimize import leastsq
from scipy.optimize import differential_evolution

_MINIMIZE = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'Anneal',
    'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']

MACHEPS = np.finfo(np.float64).eps

class FitResult(object):
    def __init__(self, p=None, cost=np.nan, cov_p=None, success=False,
                 status=-1, message='', nfev=-1, **kwds):
        self.p = np.copy(p)
        self.cost = cost
        self.message = message
        self.nfev = nfev
        self.success = success
        self.cov_p = np.copy(cov_p)
        self.pheld = None

class CurveFitter(object):
    '''Non-linear regression.

    A flexible class for curvefitting analyses, using either
    scipy.optimize.leastsq or scipy.optimize.minimize.  In comparison to
    leastsq/minimize you can specify a fitfunction instead of having to
    return residuals or the costfunction itself.  Moreover, you can hold
    parameters if they don't vary during the fit.
    The default mode of operation is minimization of `chi2` using leastsq,
    :math:`np.sum(np.power((ydata - self.model) / edata), 2)`

    There are several ways of using this class.

    1) Instantiate the class 'as' is. Here you have to supply a fitfunction
    to calculate the theoretical model. Chi2 is minimised by default.
    However, you have the option of supplying a costfunction if you wish to
    minimise a different costmetric.

    2) Alternatively you can subclass the Fitter class.
        option 1) Override the Fitter.model() method.
        If you override the Fitter.model() method, then you no longer
        have to supply a fitfunction.
        -OR-
        option 2) Override the Fitter.cost() method.
        If you override the Fitter.cost() method you no longer have to
        supply a fitfunction, or a costfunction. This method should specify
        how the cost metric varies as the fitted parameters vary.

    All fits performed using an instance of this class are stored in the history
    attribute.
    '''

    def __init__(self, xdata, ydata, func, p0, edata=None, args=(), kwds={},
                 cost_func=None):
        '''
        Initialises the data for the curve fit.

        Parameters
        ----------
        xdata : np.ndarray
        Contains the independent variables for the fit. This is not used by this
        class, other than pass it directly to the fitfunction.  Whilst xdata is
        normally rank 1, if can be rank 2 if the are several independent
        variables.

        ydata : np.ndarray
        Contains the observations corresponding to each measurement point.

        func : callable
        'fitfunction' of the form func(xdata, parameters, args=(), **kwds). The
        args tuple and kwds supplied in the construction of the CurveFitter
        object are also passed in as extra arguments to the function. You can
        use None for fitfunction _IF_ you subclass CurveFitter and provide your
        own cost method, or if you subclass the model method.

        parameters : np.ndarray
        Contains _all_ the parameters to be supplied to ``func``.

        edata : np.ndarray, optional
        Contains the measured uncertainty (s.d.) for each of the observed y data
        points. (Use None if you do not have measured uncertainty on each point)

        args : tuple, optional
        Used to pass extra arguments to CurveFitter.model(), your
        fitfunction, or costfunction.

        kwds : dict, optional
        Used to pass extra arguments to CurveFitter.model(), your
        func, or cost_func.

        cost_func : callable, optional
        If you wish to minimize a cost metric other than chisqr then supply a
        function of the form cost_func(model, data, p, *args, **kwds), where
        `model` is the data returned by `func`, data is a tuple containing the
        data (xdata, ydata, edata), p is the parameter vector and args and kwds
        were used to construct the CurveFitter object.
        This cost_func is only used if a scipy.optimize.minimize method is used
        instead of leastsq.

        Notes
        -----
        The cost method is supplied by the subset of parameters that are
        being varied by the fit. If you are only varying parameters [0, 1, 3],
        then self.p[[0, 1, 3]] is supplied.
        In contrast the model method is supplied by the entire set of parameters
        (those being held and those being varied).
        '''
        self.xdata = np.copy(xdata)
        self.ydata = np.copy(ydata)
        self.npoints = np.size(ydata, 0)
        self.weighting = False
        self.edata = np.ones(self.npoints, np.float64)

        if edata is not None:
            self.weighting = True
            self.edata = np.copy(edata)

        self.func = func

        self.p0 = p0
        self.p = np.copy(p0)
        self.ptemp = np.copy(p0)
        self.nparams = np.size(p0, 0)
        self.cost_func = cost_func

        self.args = args
        self.kwds = kwds
        self.history = []

        self.fitted_parameters = np.arange(self.nparams)

    @property
    def fit_result(self):
        '''returns the last fit performed'''
        if len(self.history):
            return self.history[-1]
        else:
            return None

    def residuals(self, p_subset=None):
        '''
        return the fit residuals, :math:`(ydata - model()) / edata`

        Parameters
        ----------

        p_subset : np.ndarray, optional
        The subset of parameters that are being fitted.  This array will have
        a size in the range [0, np.size(p0)].

        Returns
        -------
        residuals : np.ndarray
        '''
        p = self.p
        if p_subset is not None:
            self.ptemp[self.fitted_parameters] = p_subset
            p = self.ptemp

        model_data = self.model(p, *self.args, **self.kwds)

        return (self.ydata - model_data) / self.edata

    def cost(self, p_subset=None):
        '''
        The default cost function for the fit object is chisq,
        :math:`np.sum(np.power((ydata - self.model) / edata), 2)`
        If you require a different cost function provide a subclass that
        overloads this method. An alternative is to provide the costfunction
        keyword to the constructor.

        Parameters
        ----------

        p_subset : np.ndarray, optional
        The subset of parameters that are being fitted.  This array will have
        a size in the range [0, np.size(p0)].

        Returns
        -------
        chisqr : :math:`np.sum(np.power((ydata - self.model) / edata), 2)`

        '''
        if self.cost_func:
            p = self.p

            if p_subset is not None:
                self.ptemp[self.fitted_parameters] = p_subset
                p = self.ptemp

            model_data = self.model(p)

            return self.cost_func(model_data, (self.xdata, self.ydata,
                                  self.edata), p, *self.args, **self.kwds)
        else:
            residuals = self.residuals(p_subset)
            return np.nansum(np.power(residuals, 2))

    def model(self, p, *args, **kwds):
        '''
        Returns the theoretical model.

        Parameters
        ----------

        p : np.ndarray
        The parameters required for the fitfunction

        Returns
        -------
        model : np.ndarray
        The theoretical model, i.e.
        :math:`self.func(self.xdata, test_parameters, *self.args, **self.kwds)`
        '''

        return self.func(self.xdata, p, *args, **kwds)

    def fit(self, method='leastsq', pheld=None, minimizer_kwds=None):
        '''
        Start the fit.

        Parameters
        ----------

        method : str or callable, optional
        Selects the fitting algorithm.
            'leastsq' - scipy.optimize.leastsq is used for the minimization,
                with chisqr.
            callable - a function that uses the scipy.optimize.minimize
                interface.
            'Nelder-Mead'
            'Powell'
            'CG'
            'BFGS'
            'Newton-CG'
            'L-BFGS-B'
            'TNC'
            'COBYLA'
            'SLSQP'
            'dogleg'
            'trust-ncg'


        pheld : sequence, optional
        Specifies the parameter numbers to hold/fix during the fit.

        minimizer_kwds : dict, optional
        Extra parameters to pass to the selected minimizer.

        Returns
        -------
        fit_result : curvefitter.FitResult object

        Notes
        -----
        It will be necessary to pass in the extra minimizer keyword arguments
        required by the minimizer method chosen.

        If you select 'leastsq', then normal least squares is
        performed and your cost_func is ignored.
        '''
        self.fitted_parameters = np.arange(self.nparams)

        if minimizer_kwds is None:
            minimizer_kwds = {}

        if pheld is not None:
            self.fitted_parameters = np.setdiff1d(self.fitted_parameters,
                                                  pheld)
        if method == 'leastsq':
            fit_result = self._leastsquares(minimizer_kwds=minimizer_kwds)
        elif method in _MINIMIZE or callable(method):
            fit_result = self._minimze(method, minimizer_kwds=minimizer_kwds)
        else:
            raise ValueError(repr(method) + 'is not a valid argument to'
                             'scipy.optimize.minimize')

        fit_result.pheld = pheld
        if fit_result.cov_p is not None:
            if not self.weighting:
             fit_result.cov_p *= self.cost() / (self.npoints -
                                 self.fitted_parameters.size)

            cov_p = np.zeros((self.nparams, self.nparams))
            for i in range(np.size(self.fitted_parameters)):
                r = self.fitted_parameters[i]
                for j in range(0, i + 1):
                    c = self.fitted_parameters[j]
                    cov_p[r, c] = fit_result.cov_p[i, j]
            fit_result.cov_p = cov_p

        self.history.append(fit_result)
        return fit_result

    def _leastsquares(self, minimizer_kwds={}):
        minimizer_kwds['full_output'] = True
        p_subset = self.p[self.fitted_parameters]
        output = scipy.optimize.leastsq(self.residuals, p_subset,
                                        **minimizer_kwds)
        p_subset, cov_p, infodict, mesg, ier = output
        success = False
        cost = np.nan
        self.p[self.fitted_parameters] = p_subset

        if ier in [1, 2, 3, 4]:
            success = True
            cost = self.cost()

        return FitResult(p=self.p, cov_p=cov_p, nfev=infodict['nfev'],
                         cost=cost, message=mesg, success=success)

    def _minimze(self, method, minimizer_kwds={}):
        '''
        minimize cost function using scipy.optimize.minimize
        '''
        minimizer_kwds['method'] = method

        p_subset = self.p[self.fitted_parameters]

        opt_res = scipy.optimize.minimize(self.cost, p_subset, **minimizer_kwds)
        opt_res.cost = opt_res.fun

        self.p[self.fitted_parameters] = opt_res.x
        opt_res.p = self.p

        '''
        depending on the method chosen minimize may not always provide hess or
        hess_inv in opt_res.  To try and standardise the output we will use a
        single way of estimating the covariance matrix.
        '''
        opt_res.cov_p = self.estimate_covariance_matrix()

        fit_result = FitResult(**opt_res)
        return fit_result

    def estimate_covariance_matrix(self):
        '''
        Estimates the covariance matrix.
        '''
        alpha, beta = self.estimate_mrqcof(self.p[self.fitted_parameters])
        return scipy.linalg.pinv(alpha)

    def estimate_mrqcof(self, p_subset):
        '''
        Estimates the gradient and hessian matrix for the curvefit.
        '''
        nvary = self.fitted_parameters.size

        derivmatrix = np.zeros((nvary, self.npoints), np.float64)
        ei = np.zeros((nvary,), np.float64)
        epsilon = (pow(MACHEPS, 1. / 3)
                   * np.fmax(np.fabs(p_subset), 0.1))

        alpha = np.zeros((nvary, nvary), np.float64)
        beta = np.zeros(nvary, np.float64)

        #this is wasteful of function evaluations. Requires 2 evaluations for
        #LHS and RHS
        for k in range(nvary):
            self.ptemp[self.fitted_parameters] = p_subset
            d = epsilon[k]

            rcost = self.cost(p_subset[k] + d)
            lcost = self.cost(p_subset[k] - d)
            beta[k] = (rcost - lcost) / 2. / d

            self.ptemp[self.fitted_parameters[k]] = p_subset[k] + d
            f2 = self.model(self.ptemp, *self.args, **self.kwds)
            self.ptemp[self.fitted_parameters[k]] = p_subset[k] - d
            f1 = self.model(self.ptemp, *self.args, **self.kwds)

            derivmatrix[k, :] = (f2 - f1) / 2. / d

        for i in range(nvary):
            for j in range(i + 1):
                val = np.sum(derivmatrix[i] * derivmatrix[j] / self.edata**2)
                alpha[i, j] = val;
                alpha[j, i] = val;

        return alpha, beta


def de_wrapper(func, x0, args=(), **kwargs):
    '''
    A wrapper for scipy.optimize.differential_evolution that allows it to be
    used as a custom method in scipy.optimize.minimize
    '''
    de_kwds = {'strategy':'best1bin', 'maxiter':None, 'popsize':15,
               'tol':0.01, 'mutation':(0.5, 1), 'recombination':0.7,
               'seed':None, 'callback':None, 'disp':False, 'polish':True,
               'init':'latinhypercube'}

    for k, v in kwargs.items():
        if k in de_kwds:
            de_kwds[k] = v

    return differential_evolution(func, kwargs['bounds'], args=args, **de_kwds)

if __name__ == '__main__':
    def gauss(x, p, *args):
        return p[0] + p[1] * np.exp(-((x - p[2]) / p[3])**2)
    xdata = np.linspace(-4, 4, 100)
    p0 = np.array([0., 1., 0., 1.])
    ydata = gauss(xdata, p0)

    f = CurveFitter(xdata, ydata, gauss, p0 + 0.2)
    res = f.fit()
    print(res.p, np.diag(res.cov_p))

    bounds = [(-1., 1.), (0., 2.), (-3., 3.), (0.001, 2.)]
    res1 = f.fit(method=de_wrapper, minimizer_kwds={'bounds':bounds})
    print(res.p, np.diag(res1.cov_p))

