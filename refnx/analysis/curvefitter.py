# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:37:29 2014

@author: anz
"""
from __future__ import print_function
from lmfit import Minimizer, Parameters
# import pymc
import numpy as np
import numpy.ma as ma
import re
import warnings
# check for EMCEE
HAS_EMCEE = False
try:
    import emcee as emcee
    from pandas import DataFrame
    HAS_EMCEE = True
except ImportError:
    pass


_MACHEPS = np.finfo(np.float64).eps


def to_Parameters(p0, varies=None, bounds=None, names=None, expr=None):
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
        names = ['p%d' % i for i in range(p0.size)]

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
    # go through and add the parameters
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


def varys(params):
    """
    A convenience function that takes an lmfit.Parameters instance and finds
    out which ones vary

    Parameters
    ----------
    parameters : lmfit.Parameters

    Returns
    -------
    varys : bool, sequence
        Which parameters are varying
    """
    return [params[par].vary for par in params]


def exprs(params):
    """
    A convenience function that takes an lmfit.Parameters instance and returns
    the the constraint expressions

    Parameters
    ----------
    parameters : lmfit.Parameters

    Returns
    -------
    exprs : list of str

    """
    expr = [params[par].expr for par in params]
    return expr


def values(params):
    """
    A convenience function that takes an lmfit.Parameters instance and returns
    the the values
    """
    return np.array([param.value for param in params.values()], np.float64)


def names(params):
    return list(params.keys())


def bounds(params):
    return [(params[par].min, params[par].max) for par in params]


def clear_bounds(params):
    for par in params:
        params[par].min = -np.inf
        params[par].max = np.inf


def fitfunc(f):
    """
    A decorator that can be used to say if something is a fitfunc.
    """
    f.fitfuncwraps = True
    return f


class CurveFitter(Minimizer):
    """
    A curvefitting class that extends lmfit.Minimize
    """
    def __init__(self, fitfunc, xdata, ydata, params, edata=None, mask=None,
                 fcn_args=(), fcn_kws=None, kws=None, callback=None):
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
        mask : np.ndarray, optional
            A boolean array with the same shape as ydata.  If mask is True
            then that point is excluded from the residuals calculation.
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

        if mask is not None:
            if self.ydata.shape != mask.shape:
                raise ValueError('mask shape should be same as data')

            self.mask = mask
        else:
            self.mask = np.empty(self.ydata.shape, bool)
            self.mask[:] = False

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

    @property
    def data(self):
        # returns the unmasked data, and the mask
        return (self.xdata,
                self.ydata,
                self.edata,
                self.mask)

    @data.setter
    def data(self, data):
        self.xdata = np.asfarray(data[0])
        self.ydata = np.asfarray(data[1])
        if len(data) > 2:
            self.edata = np.asfarray(data[2])

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

        Note
        ----
        This method should only return the points that are not masked.
        """
        model = self.model(params)
        resid = (model - self.ydata) / self.edata

        if self.mask is not None:
            resid_ma = ma.array(resid, mask=self.mask)
            return resid_ma[~resid_ma.mask].data
        else:
            return resid

    def model(self, params):
        """
        Calculates the model.

        Parameters
        ----------
        params : lmfit.Parameters instance
            Specifies the entire parameter set

        Returns
        -------
        model : array_like
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
        result = self.minimize(method=method)
        return result

    @staticmethod
    def parameter_names(nparams=0):
        """
        Provides a set of names for constructing an lmfit.Parameters instance

        Parameters
        ----------
        nparams: int, optional
                >= 0 - provide a set of names with length `nparams`
        Returns
        -------
        names: list
            names for the lmfit.Parameters instance
        """
        name = list()
        if nparams > 0:
            name = ['p%d' % i for i in range(nparams)]
        return name

    def emcee(self, params=None, steps=1000, nwalkers=100, burn=0, thin=1,
              ntemps=1):
        """
        Samples the posterior for the curvefitting system using MCMC.
        This method updates curvefitter.params at the end of the sampling
        process.  You have to have set bounds on all of the parameters, and it
        is assumed that the prior is Uniform. You need to have `emcee` and
        `pandas` installed to use this method.

        Parameters
        ----------
        steps : int, optional
            How many samples you would like to draw from the posterior
            distribution.
        burn : int, optional
            Discard this many samples from the start of the sampling regime.
        thin : int, optional
            Only accept 1 in every `thin` samples.
        ntemps : int, optional
            If `ntemps > 1` perform a Parallel Tempering.

        Returns
        -------
        result, chain : MinimizerResult, pandas.DataFrame
            MinimizerResult object contains updated params, fit statistics, etc.
            The `chain` contains the samples. Has shape (steps, ndims).
        """
        if not HAS_EMCEE:
            raise NotImplementedError('You must have emcee to use the emcee '
                                      'method')

        result = self.prepare_fit(params=params)
        vars   = result.init_vals
        params = result.params

        # Removing internal parameter scaling. We could possibly keep it,
        # but I don't know how this affects the emcee sampling.
        bounds_varying = []
        for i, par in enumerate(params):
            param = params[par]
            vars[i] = param.from_internal(param.value)
            param.from_internal = lambda val: val
            lb, ub = param.min, param.max
            if lb is None or lb is np.nan:
                lb = -np.inf
            if ub is None or ub is np.nan:
                ub = np.inf
            bounds_varying.append((lb, ub))

        bounds_varying = np.array(bounds_varying)

        self.nvarys = len(result.var_names)

        def lnlike(theta):
            # log likelihood
            return -0.5 * self.penalty(theta)

        def lnprior(theta):
            # stay within the prior specified by the parameter bounds
            if (np.any(theta > bounds_varying[:, 1])
                    or np.any(theta < bounds_varying[:, 0])):
                return -np.inf
            return 0

        def lnprob(theta):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta)

        if ntemps > 1:
            # TODO setup pos
            sampler = emcee.PTSampler(ntemps, nwalkers, self.nvarys, lnlike,
                                      lnprior)
        else:
            p0 = np.array([vars * (1 + 1e-2 * np.random.randn(self.nvarys))
                        for i in range(nwalkers)])
            sampler = emcee.EnsembleSampler(nwalkers, self.nvarys, lnprob)

        # burn in the sampler
        for output in sampler.sample(p0, iterations=burn):
            p0 = output[0]
        sampler.reset()

        # now do a production run
        for output in sampler.sample(p0, iterations=steps - burn, thin=thin):
            pass

        flat_chain = DataFrame(sampler.flatchain, columns=result.var_names)

        mean = np.mean(flat_chain, axis=0)
        quantiles = np.percentile(flat_chain, [15.8, 84.2], axis=0)

        for i, var_name in enumerate(result.var_names):
            std_l, std_u = quantiles[:, i]
            params[var_name].value = mean[i]
            params[var_name].stderr = std_u - std_l
            params[var_name].correl = {}

        # work out correlation coefficients
        corrcoefs = np.corrcoef(flat_chain.T)

        for i, var_name in enumerate(result.var_names):
            for j, var_name2 in enumerate(result.var_names):
                if i != j:
                    result.params[var_name].correl[var_name2] = corrcoefs[i, j]

        result.errorbars == True
        result.ndata = 1
        result.nfree = 1
        result.chisqr = self.penalty(mean)
        result.redchi = result.chisqr / (result.ndata - result.nvarys)
        self.unprepare_fit()

        return result, flat_chain

    def mcmc1(self, samples=1e4, burn=0, thin=1, verbose=0):
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


class GlobalFitter(CurveFitter):
    """
    A class for simultaneous curvefitting of multiple datasets
    """
    def __init__(self, fitters, constraints=(), kws=None,
                 callback=None):
        """
        fitters: sequence of CurveFitter instances
            Contains all the fitters and fitfunctions for the global fit.
        constraints: str sequence, optional
            Of the type 'dN:param_name = constraint'. Sets a constraint
            expression for the parameter `param_name` in dataset N. The
            constraint 'd2:scale = 2 * d0:back' constrains the `scale`
            parameter in dataset 2 to be twice the `back` parameter in
            dataset 0.
        kws: dict, optional
            Extra minimization keywords to be passed to the minimizer of
            choice.
        callback: callable, optional
            Function called at each step of the minimization. Has the
            signature ``callback(params, iter, resid)``
        """
        min_kwds = {}
        if kws is not None:
            min_kwds = kws

        for fitter in fitters:
            if not isinstance(fitter, CurveFitter):
                raise ValueError('All items in curve_fitter_list must be '
                                 'instances of CurveFitter')

        self.fitters = fitters

        self.new_param_names = []
        p = Parameters()
        for i, fitter in enumerate(self.fitters):
            # add all the parameters for a given dataset
            # the parameters are all given new names:
            # abc -> abc_d0
            # parameter `abc` in dataset 0 becomes abc_d0
            new_names = {}
            for j, item in enumerate(fitter.params.items()):
                old_name = item[0]
                param = item[1]
                new_name = old_name + '_d%d' % i
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
            for param in fitter.params.values():
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

        # now set constraints/linkages up. They're specified as
        # dN:param_name=constraint
        dp_string = 'd([0-9]+):([0-9a-zA-Z_]+)'
        constraint_regex = re.compile(dp_string + '\s*=\s*(.*)')
        param_regex = re.compile(dp_string)

        all_names = p.valuesdict().keys()
        for constraint in constraints:
            r = constraint_regex.search(constraint)

            if r is not None:
                groups = r.groups()
                dataset_num = int(groups[0])
                param_name = groups[1]
                const = groups[2]

                # see if this parameter is in the list of parameters
                modified_param_name = param_name + ('_d%d' % dataset_num)
                if modified_param_name not in all_names:
                    continue

                # now search for fitters mentioned in constraint
                d_mentioned = param_regex.findall(const)
                for d in d_mentioned:
                    new_name = d[1] + ('_d%d' % int(d[0]))
                    # see if the dataset mentioned is actually a parameter
                    if new_name in self.new_param_names[int(d[0])]:
                        # if it is, then rename it.
                        const = const.replace('d' + d[0] + ':' + d[1],
                                              new_name)
                    else:
                        const = None
                        break

                p[modified_param_name].set(expr=const)

        self.params = p
        xdata = [fitter.xdata for fitter in fitters]
        ydata = [fitter.ydata for fitter in fitters]
        edata = [fitter.edata for fitter in fitters]
        super(GlobalFitter, self).__init__(None,
                                           np.hstack(xdata),
                                           np.hstack(ydata),
                                           self.params,
                                           edata=np.hstack(edata),
                                           callback=callback,
                                           kws=min_kwds)

    def distribute_params(self, params):
        """
        Takes the combined parameter set and distributes linked parameters
        into the individual parameter sets.

        Parameters
        ----------
        params: lmfit.Parameters
            Specifies the entire parameter set, across all the datasets
        """
        vals = params.valuesdict()
        for i, fitter in enumerate(self.fitters):
            new_names = self.new_param_names[i]
            for new_name, old_name in new_names.items():
                fitter.params[old_name].set(value=vals[new_name])

    def model(self, params):
        """
        Calculates the model.

        Parameters
        ----------
        params: lmfit.Parameters
            Specifies the entire parameter set, across all the datasets

        Returns
        -------
        model : np.ndarray
            The model.
        """
        model = np.zeros(0, dtype='float64')
        self.distribute_params(params)

        for fitter in self.fitters:
            model = np.append(model,
                              fitter.model(fitter.params))
        return model

    def residuals(self, params):
        """
        Calculate the difference between the data and the model. Also known as
        the objective function.  This function is minimized during a fit.
        residuals = (fitfunc - ydata) / edata

        Parameters
        ----------
        params: lmfit.Parameters
            Specifies the entire parameter set

        Returns
        -------
        residuals : np.ndarray
            The difference between the data and the model.
        """
        total_residuals = np.zeros(0, dtype='float64')
        self.distribute_params(params)

        for fitter in self.fitters:
            resid = fitter.residuals(fitter.params)
            total_residuals = np.append(total_residuals,
                                        resid)

        return total_residuals


if __name__ == '__main__':
    from lmfit import fit_report

    def gauss(x, params, *args):
        """Calculates a Gaussian model"""
        p = params.valuesdict().values()
        return p[0] + p[1] * np.exp(-((x - p[2]) / p[3])**2)

    xdata = np.linspace(-4, 4, 100)
    p0 = np.array([0., 1., 0., 1.])
    bounds = [(-1., 1.), (0., 2.), (-3., 3.), (0.001, 2.)]

    temp_pars = to_Parameters(p0, bounds=bounds)
    pars = to_Parameters(p0 + 0.2, bounds=bounds)

    ydata = gauss(xdata, temp_pars) + 0.1 * np.random.random(xdata.size)

    f = CurveFitter(gauss, xdata, ydata, pars)
    f.fit()

    print(fit_report(f.params))

    MDL = f.mcmc(samples=1e4)
    print(fit_report(f.params))