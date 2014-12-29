# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:37:29 2014

@author: anz
"""
from __future__ import print_function
from lmfit import Minimizer, Parameters
#import pymc
import numpy as np
import re
import warnings


_MACHEPS = np.finfo(np.float64).eps

def params(p0, varies=None, bounds=None, names=None, expr=None):
    """
    Utility function to convert sequences into a lmfit.Parameters instance

    Parameters
    ----------
    p0 : np.ndarray
        numpy array containing parameter values.
    varies : bool sequence, optional
        Specifies whether a parameter is being held or varied.
    bounds : sequence, optional
        Tuple of (min, max) pairs specifying the lower and upper bounds for
        each parameter
    name : str sequence, optional
        Name of each parameter
    expr : str sequence, optional
        Constraints for each parameter

    Returns
    -------
    p : lmfit.Parameters instance
    """
    if varies is None:
        _varies = [True] * p0.size
    else:
        _varies = list(varies)

    if names is None:
        names = ['p%d'%i for i in range(p0.size)]

    if bounds is not None:
        lowlim = []
        hilim = []
        for bound in bounds:
            lowlim.append(bound[0])
            hilim.append(bound[1])
    else:
        lowlim = [None] * p0.size
        hilim = [None] * p0.size

    if expr is None:
        expr = [None] * p0.size
        
    _p0 = np.copy(p0)
        
    p = Parameters()
    #go through and add the parameters
    for i in range(p0.size):
        # if the limits are finite and equal, then you shouldn't be fitting
        # the parameter. So fix the parameter and set the upper limit to be
        # slightly larger (otherwise you'll get an error when setting the
        # Parameter up)
        if (lowlim[i] is not None and hilim[i] is not None and
            np.isfinite(lowlim[i]) and np.isfinite(hilim[i]) and
            lowlim[i] == hilim[i]):
            hilim[i] += 1
            _p0[i] = lowlim[i]
            _varies[i] = False
            warnings.warn('Parameter min==max and parameter %s was varying. %s'
                          ' now fixed' % (names[i], names[i]), RuntimeWarning)
                
        p.add(names[i], value=_p0[i], min=lowlim[i], max=hilim[i],
              vary=_varies[i], expr=expr[i])

    return p


class CurveFitter(Minimizer):
    """
    A curvefitting class that extends lmfit.Minimize
    """
    def __init__(self, fitfunc, xdata, ydata, params, edata=None, fcn_args=(),
                fcn_kws=None, kws=None, callback=None):
        """
        fitfunc : callable
            Function calculating the model for the fit.  Should have the
            signature: ``fitfunc(xdata, params, *fcn_args, **fcn_kws)``
        xdata : np.ndarray
            The independent variables
        ydata : np.ndarray
            The dependent (observed) variable
        params : lmfit.Parameters instance
            Specifies the parameter set for the fit
        edata : np.ndarray, optional
            The measured uncertainty in the dependent variable, expressed as
            sd.  If this array is not specified, then edata is set to unity.
        fcn_args : tuple, optional
            Extra parameters required to fully specify fitfunc.
        fcn_kws : dict, optional
            Extra keyword parameters needed to fully specify fitfunc.
        minimizer_kwds : dict, optional
            Keywords passed to the minimizer.
        callback : callable, optional
            A function called at each minimization step. Has the signature:
            ``callback(params, iter, resid, *args, **kwds)``
        """
        self.fitfunc = fitfunc
        self.xdata = np.asfarray(xdata)
        self.ydata = np.asfarray(ydata)
        self.MDL = None
        if edata is not None:
            self.edata = np.asfarray(edata)
            self.scale_covar = False
        else:
            self.edata = np.ones_like(self.ydata)
            self.scale_covar = True

        min_kwds = {}
        if kws is not None:
            min_kwds = kws

        super(CurveFitter, self).__init__(self.residuals,
                                          params,
                                          iter_cb=callback,
                                          fcn_args=fcn_args,
                                          fcn_kws=fcn_kws,
                                          scale_covar=self.scale_covar,
                                          **min_kwds)

    def residuals(self, params, *args, **kwds):
        """
        Calculate the difference between the data and the model.
        Also known as the objective function.  This function is minimized
        during a fit.
        residuals = (fitfunc - ydata) / edata

        Parameters
        ----------
        params : lmfit.Parameters instance
            Specifies the entire parameter set

        Returns
        -------
        residuals : np.ndarray
            The difference between the data and the model.
        """
        model = self.model(params)
        resid = (model - self.ydata) / self.edata
        return resid.flatten()

    def model(self, params):
        """
        Calculates the model.

        Parameters
        ----------
        params : lmfit.Parameters instance
            Specifies the entire parameter set

        Returns
        -------
        model : np.ndarray
            The model.
        """
        return self.fitfunc(self.xdata, params, *self.userargs,
                            **self.userkws)

    def fit(self, method='leastsq'):
        """
        Fits the dataset.

        Parameters
        ----------
        method : str, optional
            Name of the fitting method to use.
            One of:
            'leastsq'                -    Levenberg-Marquardt (default)
            'nelder'                 -    Nelder-Mead
            'lbfgsb'                 -    L-BFGS-B
            'powell'                 -    Powell
            'cg'                     -    Conjugate-Gradient
            'newton'                 -    Newton-CG
            'cobyla'                 -    Cobyla
            'tnc'                    -    Truncate Newton
            'trust-ncg'              -    Trust Newton-CGn
            'dogleg'                 -    Dogleg
            'slsqp'                  -    Sequential Linear Squares Programming
            'differential_evolution' -    differential evolution

        Returns
        -------
        success : bool
            Whether the fit succeeded.
        """
        return self.minimize(method=method)
    
    def mcmc(self, samples=1e4, burn=0, thin=1, verbose=0):
        """
        Samples the posterior for the curvefitting system using MCMC.
        This method updates curvefitter.params at the end of the sampling
        process.  You have to have set bounds on all of the parameters, and it
        is assumed that the prior is Uniform.
        
        Parameters
        ----------
        samples : int, optional
            How many samples you would like to draw from the posterior
            distribution.
        burn : int, optional
            Discard this many samples from the start of the sampling regime.
        thin : int, optional
            Only accept 1 in `thin` samples.
        verbose : integer, optional
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
        
        Returns
        -------
        MDL : pymc.MCMC.MCMC instance
            Contains the samples.
        """
        return super(CurveFitter, self).mcmc(samples, burn=burn, thin=thin)

#==============================================================================
#         # fitted is a dict of tuples. the key is the param name. The tuple
#         # (i, j) has i = i'th parameter, j = index into the j'th fitted
#         # parameter
#         fitted = {}
#         self.__fun_evals = 0
#         j = 0
#         for i, par in enumerate(self.params):
#             parameter = self.params[par]
#             if parameter.vary:
#                 fitted[parameter.name] = (i, j)
#                 j += 1
#         
#         def driver():
#             p = np.empty(len(fitted), dtype=object)
#             for par, idx in fitted.items():
#                 parameter = self.params[par]
#                 p[idx[1]] = pymc.Uniform(parameter.name, parameter.min,
#                                          parameter.max, value=parameter.value)
#     
#             @pymc.deterministic(plot=False)
#             def model(p=p):
#                 self.__fun_evals += 1
#                 for name in fitted:
#                     self.params[name].value = p[fitted[name][1]]
#                 return self.model(self.params)
# 
#             y = pymc.Normal('y', mu=model, tau=1.0 / self.edata**2,
#                             value=self.ydata, observed=True)
# 
#             return locals()
# 
#         MDL = pymc.MCMC(driver(), verbose=verbose)
#         MDL.sample(samples, burn=burn, thin=thin)
#         stats = MDL.stats()
# 
#         #work out correlation coefficients
#         corrcoefs = np.corrcoef(np.vstack(
#                      [MDL.trace(par, chain=None)[:] for par in fitted.keys()]))
#                 
#         for par in self.params:
#             self.params[par].stderr = None
#             self.params[par].correl = None
# 
#         for par in fitted.keys():
#             i = fitted[par][1]
#             param = self.params[par]
#             param.correl = {}
#             param.value = stats[par]['mean']
#             param.stderr = stats[par]['standard deviation']
#             for par2 in fitted.keys():
#                 j = fitted[par2][1]
#                 if i != j:
#                     param.correl[par2] = corrcoefs[i, j]
# 
#         self.MDL = MDL
#         self.ndata = self.ydata.size
#         self.nvarys = len(fitted)
#         self.nfev = self.__fun_evals
#         self.chisqr = np.sum(self.residuals(self.params) ** 2)
#        self.redchi = self.chisqr / (self.ndata - self.nvarys)
#        del(self.__fun_evals)
#        return MDL
#==============================================================================


class GlobalFitter(CurveFitter):
    """
    A class for simultaneous curvefitting of multiple datasets
    """
    def __init__(self, datasets, constraints=(), kws=None,
                 callback=None):
        """
        datasets : sequence of CurveFitter instances
            Contains all the datasets and fitfunctions for the global fit.
        constraints : str sequence, optional
            Of the type 'dNpM:constraint'. Sets a constraint expression for
            parameter M in dataset N.  The constraint 'd2p3:d0p1' constrains
            parameter 3 in dataset 2 to be equal to parameter 1 in dataset 0.
        kws : dict, optional
            Extra minimization keywords to be passed to the minimizer of
            choice.
        callback : callable, optional
            Function called at each step of the minimization. Has the signature
            ``callback(params, iter, resid)``

        """
        min_kwds = {}
        if kws is not None:
            min_kwds = kws

        for dataset in datasets:
            if not isinstance(dataset, CurveFitter):
                raise ValueError('All items in curve_fitter_list must be '
                                 'instances of CurveFitter')

        self.datasets = datasets

        self.new_param_names = []
        p = Parameters()
        for i, dataset in enumerate(self.datasets):
            # add all the parameters for a given dataset
            new_names = {}
            for j, item in enumerate(dataset.params.items()):
                old_name = item[0]
                param = item[1]
                new_name = old_name + '_d%dp%d' % (i, j)
                new_names[new_name] = old_name

                p.add(new_name,
                      value=param.value,
                      vary=param.vary,
                      min=param.min,
                      max=param.max,
                      expr=param.expr)

            self.new_param_names.append(new_names)

            # if there are any expressions they have to be updated
            # iterate through all the parameters in the dataset
            old_names = dict((v, k) for k, v in new_names.items())
            for i, param in enumerate(dataset.params.values()):
                expr = param.expr
                new_name = old_names[param.name]
                # if it's got an expression you'll have to update it
                if expr is not None:
                    # see if any of the old names are in there.
                    for old_name in old_names:
                        regex = re.compile('(%s)' % old_name)
                        if regex.search(expr):
                            new_expr_name = old_names[old_name]
                            new_expr = expr.replace(old_name, new_expr_name)
                            p[new_name].set(expr=new_expr)

        # now set constraints/linkages up. They're specified as dNpM:constraint
        regex = re.compile('d([0-9]*)p([0-9]*)[\s]*:[\s]*(.*)')
        data_regex = re.compile('d([0-9]*)p([0-9]*)')

        all_names = p.valuesdict().keys()
        for constraint in constraints:
            r = regex.search(constraint)
            if r:
                N, M = int(r.groups()[0]), int(r.groups()[1])
                const = r.groups()[2]

                n_left = re.compile('.*(_d%dp%d)' % (N, M))
                # search all the names
                name_left = [m.group(0) for l in all_names for m in
                              [n_left.search(l)] if m]
                if not len(name_left):
                    continue

                # now search for datasets mentioned in constraint
                d_mentioned = data_regex.findall(const)
                for d in d_mentioned:
                    # we need to replace all datasets mentioned in constraint
                    n_right = re.compile('.*(_d%sp%s)' % (d[0], d[1]))
                    name_right = [m.group(0) for l in all_names for m in
                                  [n_right.search(l)] if m]

                    if len(name_right):
                        const = const.replace(
                                    'd%sp%s' % (d[0], d[1]),
                                    name_right[0])

                p[name_left[0]].set(expr=const)

        self.params = p
        xdata = [dataset.xdata for dataset in datasets]
        ydata = [dataset.ydata for dataset in datasets]
        edata = [dataset.edata for dataset in datasets]
        super(GlobalFitter, self).__init__(None,
                                           xdata,
                                           ydata,
                                           self.params,
                                           edata=edata,
                                           callback=callback,
                                           kws=min_kwds)

    def model(self, params):
        """
        Calculates the model.

        Parameters
        ----------
        params : lmfit.Parameters instance
            Specifies the entire parameter set, across all the datasets

        Returns
        -------
        model : np.ndarray
            The model.
        """
        values = params.valuesdict()
        model = np.zeros(0, dtype='float64')
        for i, dataset in enumerate(self.datasets):
            new_names = self.new_param_names[i]
            for new_name, old_name in new_names.items():
                dataset.params[old_name].set(value=values[new_name])
            model = np.append(model,
                              dataset.model(dataset.params))

        return model

    def residuals(self, params):
        """
        Calculate the difference between the data and the model.
        Also known as the objective function.  This function is minimized
        during a fit.
        residuals = (fitfunc - ydata) / edata

        Parameters
        ----------
        params : lmfit.Parameters instance
            Specifies the entire parameter set

        Returns
        -------
        residuals : np.ndarray
            The difference between the data and the model.
        """
        values = params.valuesdict()
        total_residuals = np.zeros(0, dtype='float64')
        for i, dataset in enumerate(self.datasets):
            new_names = self.new_param_names[i]
            for new_name, old_name in new_names.items():
                dataset.params[old_name].set(value=values[new_name])
            resid = (dataset.ydata - dataset.model(dataset.params))
            resid /= dataset.edata
            
            total_residuals = np.append(total_residuals,
                                        resid)

        return total_residuals


if __name__ == '__main__':
    from lmfit import fit_report
   
    def gauss(x, params, *args):
        'Calculates a Gaussian model'
        p = params.valuesdict().values()
        return p[0] + p[1] * np.exp(-((x - p[2]) / p[3])**2)

    xdata = np.linspace(-4, 4, 100)
    p0 = np.array([0., 1., 0., 1.])
    bounds = [(-1., 1.), (0., 2.), (-3., 3.), (0.001, 2.)]

    temp_pars = params(p0, bounds=bounds)
    pars = params(p0 + 0.2, bounds=bounds)

    ydata = gauss(xdata, temp_pars) + 0.1 * np.random.random(xdata.size)

    f = CurveFitter(gauss, xdata, ydata, pars)
    f.fit()

    print(fit_report(f.params))

#    g = GlobalFitter([f], ['d0p3:1'])
#    g.fit()
#    print(fit_report(g))

    MDL = f.mcmc(samples=1e4)
    print(fit_report(f.params))