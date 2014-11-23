from __future__ import division
import numpy as np
import math
import scipy
from scipy.optimize import leastsq
from scipy.optimize import differential_evolution

_MINIMIZE = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'Anneal',
    'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']

class FitResult(object):
    def __init__(self, p=None, cost=np.nan, cov_p=None, success=False,
                 status=-1, message='', nfev=-1, **kwds):
        self.p = p
        self.cost = cost
        self.message = message
        self.nfev = nfev
        self.success = success
        self.cov_p = cov_p
        self.pheld = None

class Fitter(object):
    def __init__(self, xdata, ydata, func, p0, edata=None, args=(), kwds={},
                 cost_func=None):

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
        if len(self.history):
            return self.history[-1]
        else:
            return None

    def residuals(self, p_subset=None):
        p = self.p
        if p_subset is not None:
            self.ptemp[self.fitted_parameters] = p_subset
            p = self.ptemp

        model_data = self.model(p, *self.args, **self.kwds)

        return (self.ydata - model_data) / self.edata

    def cost(self, p_subset=None):
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
        return self.func(self.xdata, p, *args, **kwds)

    def fit(self, method='leastsq', pheld=None, minimizer_kwds={}):
        self.fitted_parameters = np.arange(self.nparams)
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
        if fit_result.cov_p:
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

        if not self.weighting:
            cov_p *= self.cost() / (self.npoints - self.fitted_parameters.size)

        if ier in [1, 2, 3, 4]:
            success = True
            cost = self.cost()

        return FitResult(p=self.p, cov_p=cov_p, nfev=infodict['nfev'],
                         cost=cost, message=mesg, success=success)

    def _minimze(self, method, minimizer_kwds={}):
        minimizer_kwds['method'] = method
        p_subset = self.p[self.fitted_parameters]

        opt_res = scipy.optimize.minimize(self.cost, p_subset, **minimizer_kwds)
        opt_res.cost = opt_res.fun

        self.p[self.fitted_parameters] = opt_res.x
        opt_res.p = self.p
        opt_res.fitted_parameters = self.fitted_parameters

        fit_result = FitResult(**opt_res)
        return fit_result


def de_wrapper(func, x0, args=(), **kwargs):
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

    f = Fitter(xdata, ydata, gauss, p0 + 0.2)
    res = f.fit()
    print(res.p)
