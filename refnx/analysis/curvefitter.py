# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:37:29 2014

@author: Andrew Nelson
"""
from __future__ import print_function
import re
from functools import partial
import abc
import warnings
from lmfit import Minimizer, Parameters
import numpy as np
import numpy.ma as ma
from refnx.dataset import Data1D

# check for EMCEE
HAS_EMCEE = False
try:
    import emcee as emcee
    HAS_EMCEE = True
except ImportError:
    pass


_MACHEPS = np.finfo(np.float64).eps


def to_parameters(p0, varies=None, bounds=None, names=None, expr=None):
    """
    Utility function to convert sequences into a :class:`lmfit.parameter.Parameters` instance

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
        if (lowlim[i] is not None
            and hilim[i] is not None
            and np.isfinite(lowlim[i])
            and np.isfinite(hilim[i])
            and lowlim[i] == hilim[i]):

            hilim[i] += 1
            _p0[i] = lowlim[i]
            _varies[i] = False

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
    varys: bool, sequence
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
    the values
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


class FitFunction(object):
    """
    An abstract FitFunction class.
    """
    def __init__(self):
        pass

    def __call__(self, x, params, *args, **kws):
        """
        Calculate the FitFunction
        """
        return self.model(x, params, *args, **kws)

    @abc.abstractmethod
    def model(self, x, params, *args, **kws):
        """
        Calculate the predictive model for the fit.
        Override this method in your own fitfunction.

        Parameters
        ----------
        x : array-like
            The independent variable for the fit
        params : lmfit.Parameters
            The model parameters

        Returns
        -------
        predictive : np.ndarray
            The predictive model for the fitfunction.

        Notes
        -----
        `args` and `kws` can be used to fully specify the fit function.
        Normally you would supply these via when the **FitFunction** object is
        constructed.
        """
        raise RuntimeError("You can't use the abstract base FitFunction in a"
                           " real fit")

    @staticmethod
    def parameter_names(nparams=0):
        """
        Provides a set of names for constructing an :class:`lmfit.parameter.Parameters` instance

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


class CurveFitter(Minimizer):
    r"""
    A curvefitting class that extends :class:`lmfit:Minimizer.Minimizer`

Parameters
----------

fitfunc : callable
    Function calculating the generative model for the fit.  Should have
    the signature: ``fitfunc(x, params, *fcn_args, **fcn_kws)``. You
    can also supply a :class:`FitFunction` instance.
data : sequence, :class:`refnx.dataset.Data1D` instance, str or file-like object
    A sequence containing the data to be analysed.
    If `data` is a sequence then:

        * data[0] - the independent variable (x-data)

        * data[1] - the dependent (observed) variable (y-data)

        * data[2] - measured uncertainty in the dependent variable,
            expressed as a standard deviation.

    Only data[0] and data[1] are required, data[2] is optional. If data[2]
    is not specified then the measured uncertainty is set to unity.

    `data` can also be a :class:`refnx.dataset.Data1D` instance containing the data.
    If `data` is a string, or file-like object then the string or file-like
    object refers to a file containing the data. The data will be loaded
    through the :class:`refnx.dataset.Data1D` constructor.
params : :class:`lmfit.parameter.Parameters` instance
    Specifies the parameter set for the fit
mask : np.ndarray, optional
    A boolean array with the same shape as `y`.  If `mask is True`
    then that point is excluded from the residuals calculation.
fcn_args : tuple, optional
    Extra parameters required to fully specify fitfunc.
fcn_kws : dict, optional
    Extra keyword parameters needed to fully specify fitfunc.
kws : dict, optional
    Keywords passed to the minimizer.
callback : callable, optional
    A function called at each minimization step. Has the signature:
    ``callback(params, iter, resid, *args, **kwds)``
costfun : callable, optional
    specifies your own cost function to minimize. Has the signature:
    ``costfun(pars, generative, y, e, *fcn_args, **fcn_kws)`` where `pars`
    is a `lmfit.Parameters` instance, `generative` is an array returned by
    `fitfunc`, and `y` and `e` correspond to the `data[1]` and
    `data[2]` arrays. `costfun` should return a single value. See Notes for
    further details.
lnpost : callable, optional
    specifies your own log-posterior probablility function. This is only
    relevant applies to the `emcee` method. Has the signature:
    ``lnpost(pars, generative, y, e, *fcn_args, **fcn_kws)`` where `pars`
    is a `lmfit.Parameters` instance, `generative` is an array returned by
    `fitfunc`, and `y` and `e` correspond to the `data[1]` and
    `data[2]` arrays. `lnpost` should return a single float value. See
    :meth:`CurveFitter.emcee` for further details.

Notes
-----
The default cost function for CurveFitter is:

.. math::
    \chi^2=\sum \left(\frac{\mathrm{data[1]} - \mathrm{fitfunc}}{\mathrm{data[2]}}\right)^2

This user defined cost function can be used to specify other cost
functions for `differential_evolution`, `leastsq`, `least_squares`.

.. _lmfit.Minimizer: http://lmfit.github.io/lmfit-py/fitting.html#module-Minimizer
    """

    def __init__(self, fitfunc, data, params, mask=None,
                 fcn_args=(), fcn_kws=None, kws=None, callback=None,
                 costfun=None, lnpost=None):
        self.fitfunc = fitfunc
        self.costfun = costfun
        self.lnpost = lnpost
        self._cf_userargs = fcn_args
        self._cf_userkws = fcn_kws

        if isinstance(data, Data1D):
            self.xdata, self.ydata, self.edata, temp = data.data
        elif type(data) == 'str' or hasattr(data, 'seek'):
            tdata = Data1D(data)
            self.xdata, self.ydata, self.edata, temp = tdata.data
        elif len(data) == 2:
            self.xdata, self.ydata = data
            self.edata = np.zeros((0))
        elif len(data) == 3:
            self.xdata, self.ydata, self.edata = data
        else:
            raise ValueError("Couldn't decipher what kind of data"
                             " you were providing.")

        # have uncertainties have been supplied for each of the data points?
        self.scale_covar = False
        if not self.edata.size:
            self.edata = np.ones_like(self.ydata)
            self.scale_covar = True

        if mask is not None:
            if self.ydata.shape != mask.shape:
                raise ValueError('mask shape should be same as data')

            self.mask = mask
        else:
            self.mask = None

        min_kwds = {}
        if kws is not None:
            min_kwds = kws

        # setup the residual calculator
        self._update_resid()

        super(CurveFitter, self).__init__(self._resid,
                                          params,
                                          iter_cb=callback,
                                          scale_covar=self.scale_covar,
                                          **min_kwds)

    def _update_resid(self):
        """
        Updates the _resid attribute, which is a function that
        evaluates the residuals and model for the system. This
        method exists because people could update the data after
        creation of the CurveFitter object.
        """
        self._resid = partial(_parallel_residuals_calculator,
                              fitfunc=self.fitfunc,
                              data_tuple=(self.xdata,
                                          self.ydata,
                                          self.edata),
                              mask=self.mask,
                              fcn_args=self._cf_userargs,
                              fcn_kws=self._cf_userkws,
                              costfun=self.costfun)
        self.userfcn = self._resid

    @property
    def data(self):
        """
        The unmasked data, and the mask

        Returns
        -------
        (x, y, e, mask) : data tuple
        """
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
            self.scale_covar = False
        else:
            self.edata = np.ones_like(self.ydata)
            self.scale_covar = True

    def residuals(self, params=None):
        """
        Calculate the difference between the data and the model. Also known as
        the objective function. This is a convenience method. Over-riding it
        will not change a fit.

        :math:`residuals = (fitfunc - y) / edata`

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
        if params is None:
            params = self.params

        params.update_constraints()
        self._update_resid()
        return self._resid(params)

    def model(self, params=None):
        """
        Calculates the model. This is a convenience method. Over-riding it will
        not change a fit.

        Parameters
        ----------
        params : lmfit.Parameters instance
            Specifies the entire parameter set

        Returns
        -------
        model : array_like
            The model.
        """
        if params is None:
            params = self.params

        params.update_constraints()
        self._update_resid()
        return self._resid(params, model=True)

    def fit(self, method='leastsq', params=None, **kws):
        """
        Fits the dataset.

        Parameters
        -----------
        method : str, optional
            Name of the fitting method to use.
            One of:

            - 'leastsq'                -    Levenberg-Marquardt (default)
            - 'nelder'                 -    Nelder-Mead
            - 'lbfgsb'                 -    L-BFGS-B
            - 'powell'                 -    Powell
            - 'cg'                     -    Conjugate-Gradient
            - 'newton'                 -    Newton-CG
            - 'cobyla'                 -    Cobyla
            - 'tnc'                    -    Truncate Newton
            - 'trust-ncg'              -    Trust Newton-CGn
            - 'dogleg'                 -    Dogleg
            - 'slsqp'                  -    Sequential Linear Squares Programming
            - 'differential_evolution' -    differential evolution

        params : Parameters, optional
            parameters to use as starting values

        Returns
        --------
        result : lmfit.MinimizerResult
            Result object.
        """
        self._update_resid()
        result = self.minimize(method=method, params=params, **kws)
        self.params = result.params
        return result

    def emcee(self, params=None, steps=1000, nwalkers=100, burn=0, thin=1,
              ntemps=1, pos=None, reuse_sampler=False, workers=1, seed=None):
        r"""
        Bayesian sampling of the posterior distribution for the parameters
        using the `emcee` Markov Chain Monte Carlo package. By default the
        method assumes that the prior is Uniform. To implement non-uniform
        priors use the `lnpost` kwd when constructing the CurveFitter. You
        need to have `emcee` installed to use this method.

        Parameters
        ----------

        params : lmfit.Parameters, optional
            Parameters to use as starting point. If this is not specified
            then the Parameters used to initialise the CurveFitter object are
            used.
        steps : int, optional
            How many samples you would like to draw from the posterior
            distribution for each of the walkers?
        nwalkers : int, optional
            Should be set so :math:`nwalkers >> nvarys`, where `nvarys` are
            the number of parameters being varied during the fit.
            "Walkers are the members of the ensemble. They are almost like
            separate Metropolis-Hastings chains but, of course, the proposal
            distribution for a given walker depends on the positions of all
            the other walkers in the ensemble." - from the `emcee` webpage.
        burn : int, optional
            Discard this many samples from the start of the sampling regime.
        thin : int, optional
            Only accept 1 in every `thin` samples.
        ntemps : int, optional
            If `ntemps > 1` perform a Parallel Tempering.
        pos : np.ndarray, optional
            Specify the initial positions for the sampler.  If `ntemps == 1`
            then `pos.shape` should be `(nwalkers, nvarys)`. Otherwise,
            `(ntemps, nwalkers, nvarys)`. You can also initialise using a
            previous chain that had the same `ntemps`, `nwalkers` and
            `nvarys`. Note that `nvarys` may be one larger than you expect it
            to be if your `userfcn` returns an array and `is_weighted is
            False`.
        reuse_sampler : bool, optional
            If you have already run `emcee` on a given `Minimizer` object then
            it possesses an internal ``sampler`` attribute. You can continue to
            draw from the same sampler (retaining the chain history) if you set
            this option to `True`. Otherwise a new sampler is created. The
            `nwalkers`, `ntemps`, `pos`, and `params` keywords are ignored with
            this option.
            **Important**: the Parameters used to create the sampler must not
            change in-between calls to `emcee`. Alteration of Parameters
            would include changed ``min``, ``max``, ``vary`` and ``expr``
            attributes. This may happen, for example, if you use an altered
            Parameters object and call the `minimize` method in-between calls
            to `emcee`.
        workers : Pool-like or int, optional
            For parallelization of sampling.  It can be any Pool-like object
            with a map method that follows the same calling sequence as the
            built-in `map` function. If int is given as the argument, then a
            multiprocessing-based pool is spawned internally with the
            corresponding number of parallel processes. 'mpi4py'-based
            parallelization and 'joblib'-based parallelization pools can also
            be used here. **Note**: because of multiprocessing overhead it may
            only be worth parallelising if the objective function is expensive
            to calculate, or if there are a large number of objective
            evaluations per step (`ntemps * nwalkers * nvarys`).
        seed : int or `np.random.RandomState`, optional
            If `seed` is an int, a new `np.random.RandomState` instance is used,
            seeded with `seed`.
            If `seed` is already a `np.random.RandomState` instance, then that
            `np.random.RandomState` instance is used.
            Specify `seed` for repeatable minimizations.

        Returns
        -------
        result: MinimizerResult
            MinimizerResult object containing updated params, statistics,
            etc. The `MinimizerResult` also contains the ``chain``,
            ``flatchain`` and ``lnprob`` attributes. The ``chain``
            and ``flatchain`` attributes contain the samples and have the shape
            `(nwalkers, (steps - burn) // thin, nvarys)` or
            `(ntemps, nwalkers, (steps - burn) // thin, nvarys)`,
            depending on whether Parallel tempering was used or not.
            `nvarys` is the number of parameters that are allowed to vary.
            The ``flatchain`` attribute is a `pandas.DataFrame` of the
            flattened chain, `chain.reshape(-1, nvarys)`. To access flattened
            chain values for a particular parameter use
            `result.flatchain[parname]`. The ``lnprob`` attribute contains the
            log probability for each sample in ``chain``. The sample with the
            highest probability corresponds to the maximum likelihood estimate.

        Notes
        -----

        This method samples the posterior distribution of the parameters using
        Markov Chain Monte Carlo.  To do so it needs to calculate the
        log-posterior probability of the model parameters, `F`, given the data,
        `D`, :math:`\ln p(F_{true} | D)`. This 'posterior probability' is
        calculated as:

        .. math::

            \ln p(F_{true} | D) \propto \ln p(D | F_{true}) + \ln p(F_{true})

        where :math:`\ln p(D | F_{true})` is the 'log-likelihood' and
        :math:`\ln p(F_{true})` is the 'log-prior'. The default log-prior
        encodes prior information already known about the model. This method
        assumes that the log-prior probability is `-np.inf` (impossible) if the
        one of the parameters is outside its limits. The log-prior probability
        term is zero if all the parameters are inside their bounds (known as a
        uniform prior). The default log-likelihood function is given by [1]_:

        .. math::

            \ln p(D|F_{true}) = -\frac{1}{2}\sum_n \left[\frac{\left(g_n(F_{true}) - D_n \right)^2}{s_n^2}+\ln (2\pi s_n^2)\right]

        The first summand in the square brackets represents the residual for a
        given datapoint (:math:`g` being the generative model) . This term
        represents :math:`\chi^2` when summed over all datapoints.

        It is also possible to calculate your own log-posterior probability, by
        constructing the CurveFitter object with a `lnpost` function. This
        will allow you to use non-uniform priors, etc. The `lnpost` function
        has the signature:
        ``lnpost(pars, generative, y, e, *fcn_args, **fcn_kws)``. You should
        return a single float from this `lnpost` function

        References
        ----------
        .. [1] http://dan.iel.fm/emcee/current/user/line/

        """

        self._update_resid()

        try:
            if self.is_weighted or self.lnpost is not None:
                # get the proper log-likelihood if you have
                # uncertainties
                self._resid = partial(_parallel_likelihood_calculator,
                                      fitfunc=self.fitfunc,
                                      data_tuple=(self.xdata,
                                                  self.ydata,
                                                  self.edata),
                                      mask=self.mask,
                                      fcn_args=self._cf_userargs,
                                      fcn_kws=self._cf_userkws,
                                      lnpost=self.lnpost
                                      )
                self.userfcn = self._resid
            else:
                pass

            result = super(CurveFitter, self).emcee(params=params, steps=steps,
                                                    nwalkers=nwalkers,
                                                    burn=burn,
                                                    thin=thin, ntemps=ntemps,
                                                    pos=pos,
                                                    reuse_sampler=reuse_sampler,
                                                    workers=workers,
                                                    float_behavior='posterior',
                                                    is_weighted=self.is_weighted,
                                                    seed=seed)
        finally:
            self._update_resid()

        self.params = result.params
        return result

    @property
    def is_weighted(self):
        """
        Returns the truth that the fit is weighted by measurement uncertainties
        """
        return not self.scale_covar

    def _resampleMC(self, samples, method='differential_evolution',
                    params=None):
        """
        Monte Carlo Resampling. Refits synthesised data `samples` times. Each
        synthesised dataset is created from the original dataset by adding
        Gaussian based on the size of the datapoint:
        ``synth = y + e * np.random.randn(y.size)``
        The parameters from each of these fits should be distributed in a way
        that is related to their statistical uncertainty.

        Parameters
        ----------
        samples : int
            Number of synthesis/refit cycles.
        method : str
            Minimisation method. See the `fit` method for other options.

        Returns
        -------
        result : lmfit.MinimizerResult
            Result object.

        Notes
        -----
        This method is currently semi-private and may disappear in future
        releases. The uncertainties are estimated as half of the 15.87, 84.13
        percentiles. The parameter value is estimated as the 50 th percentile.
        The `result` instance also contains the `mc` attribute which is an
        array the contains the result of each sample. This array has shape
        `(samples, len(result.var_names))` (i.e. only the varying parameters
        are given).
        """
        # data does
        if self.scale_covar:
            raise ValueError("To MC resample the data has to have errorbars")

        x, y, e, m = self.data

        tparams = params
        output = self.prepare_fit(params=tparams)
        params = output.params

        try:
            mc = np.zeros((samples, len(output.var_names)))
            for idx in range(samples):
                # synthesize a dataset
                ne = y + e * np.random.randn(y.size)
                self.ydata = ne

                # update the _resid attribute
                self._update_resid()

                # do a fit
                res = self.fit(method=method, params=params)

                # append values from fit
                for idx2, var_name in enumerate(output.var_names):
                    mc[idx, idx2] = res.params[var_name].value
        finally:
            self.ydata = y

        quantiles = np.percentile(mc, [15.87, 50, 84.13], axis=0)

        for i, var_name in enumerate(output.var_names):
            std_l, median, std_u = quantiles[:, i]
            params[var_name].value = median
            params[var_name].stderr = 0.5 * (std_u - std_l)
            params[var_name].correl = {}

        params.update_constraints()

        # work out correlation coefficients
        corrcoefs = np.corrcoef(mc.T)

        for i, var_name in enumerate(output.var_names):
            for j, var_name2 in enumerate(output.var_names):
                if i != j:
                    output.params[var_name].correl[var_name2] = corrcoefs[i, j]

        output.mc = mc
        output.errorbars = True
        output.nvarys = len(output.var_names)

        return output


class GlobalFitter(CurveFitter):
    """
    Simultaneous curvefitting of multiple datasets

    fitters : sequence of :class:`refnx.analysis.curvefitter.CurveFitter` instances
        Contains all the fitters and fitfunctions for the global fit.
    constraints : str sequence, optional
        Of the type 'dN:param_name = constraint'. Sets a constraint
        expression for the parameter `param_name` in dataset N. The
        constraint 'd2:scale = 2 * d0:back' constrains the `scale`
        parameter in dataset 2 to be twice the `back` parameter in
        dataset 0.
        **Important** For a parameter (`d2:scale` in this example) to be
        constrained by this mechanism it must not have any pre-existing
        constraints within its individual fitter. If there are pre-existing
        constraints then those are honoured, and constraints specified here are
        ignored.
    kws : dict, optional
        Extra minimization keywords to be passed to the minimizer of choice.
    callback : callable, optional
        Function called at each step of the minimization. Has the signature
        ``callback(params, iter, resid)``
    """
    def __init__(self, fitters, constraints=(), kws=None,
                 callback=None):

        min_kwds = {}
        if kws is not None:
            min_kwds = kws

        for fitter in fitters:
            if not isinstance(fitter, CurveFitter):
                raise ValueError('All items in curve_fitter_list must be '
                                 'instances of CurveFitter')

        self.fitters = fitters

        # new_param_reference.keys() will be the parameter names for the
        # composite fitting problem. new_param_reference.values() are the
        # (i, original_name) of the individual Parameter(s) from the individual
        # fitting problem, where i is the index of the fitter it's in.
        self.new_param_reference = dict()
        p = Parameters()
        for i, fitter in enumerate(self.fitters):
            # add all the parameters for a given dataset
            # the parameters are all given new names:
            # abc -> abc_d0
            # parameter `abc` in dataset 0 becomes abc_d0
            new_names = {}
            for old_name, param in fitter.params.items():
                new_name = old_name + '_d%d' % i
                new_names[new_name] = old_name

                self.new_param_reference[new_name] = (i, param.name)

                p.add(new_name,
                      value=param.value,
                      vary=param.vary,
                      min=param.min,
                      max=param.max,
                      expr=param.expr)

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
                            p[new_name].expr = new_expr

        # now set constraints/linkages up. They're specified as
        # dN:param_name = constraint
        dp_string = 'd([0-9]+):([0-9a-zA-Z_]+)'
        parameter_regex = re.compile(dp_string + '\s*=\s*(.*)')
        constraint_regex = re.compile(dp_string)

        for constraint in constraints:
            r = parameter_regex.search(constraint)

            if r is not None:
                groups = r.groups()
                dataset_num = int(groups[0])
                param_name = groups[1]
                const = groups[2]

                # see if this parameter is in the list of parameters
                par_to_be_constrained = param_name + ('_d%d' % dataset_num)
                if par_to_be_constrained not in p:
                    continue

                # if it already has a constraint / expr don't override it
                if p[par_to_be_constrained].expr is not None:
                    warning_msg = ("%s already has a constraint within its"
                                   " individual fitter. The %s constraint"
                                   "is ignored."
                                   % (par_to_be_constrained, constraint))
                    warnings.warn(warning_msg, UserWarning)
                    continue

                # now search for fitters mentioned in constraint
                d_mentioned = constraint_regex.findall(const)
                for d in d_mentioned:
                    new_name = d[1] + ('_d%d' % int(d[0]))
                    # see if the dataset mentioned is actually a parameter
                    if new_name in self.new_param_reference:
                        # if it is, then rename it.
                        const = const.replace('d' + d[0] + ':' + d[1],
                                              new_name)
                    else:
                        const = None
                        break

                p[par_to_be_constrained].expr = const

        self.params = p
        xdata = [fitter.xdata for fitter in fitters]
        ydata = [fitter.ydata for fitter in fitters]
        edata = [fitter.edata for fitter in fitters]

        original_params = [fitter.params for fitter in fitters]
        original_userargs = [fitter.userargs for fitter in fitters]
        original_kws = [fitter.userkws for fitter in fitters]

        self._fitfunc = partial(_parallel_global_fitfunc,
                fitfuncs=[fitter.fitfunc for fitter in fitters],
                new_param_reference=self.new_param_reference,
                original_params=original_params,
                original_userargs=original_userargs,
                original_kws=original_kws)

        super(GlobalFitter, self).__init__(self._fitfunc,
                                           (xdata, np.hstack(ydata), np.hstack(edata)),
                                           self.params,
                                           callback=callback,
                                           kws=min_kwds)

    def model(self, params=None):
        """
        Calculates the model. This method is provided for convenience purposes
        and is not used during a fit.

        Parameters
        ----------
        params: lmfit.Parameters
            Specifies the entire parameter set, across all the datasets

        Returns
        -------
        model : np.ndarray
            The model.
        """
        if params is None:
            params = self.params

        params.update_constraints()
        return self._fitfunc(self.xdata, params=params)

    def residuals(self, params=None):
        """
        Calculate the difference between the data and the model. Also known as
        the objective function.  This is a convenience method. Over-riding it
        does not change the fitting process.
        residuals = (fitfunc - y) / edata

        Parameters
        ----------
        params: lmfit.Parameters
            Specifies the entire parameter set

        Returns
        -------
        residuals : np.ndarray
            The difference between the data and the model.
        """
        if params is None:
            params = self.params

        params.update_constraints()
        return super(GlobalFitter, self).residuals(params)

    def distribute_params(self, params):
        """
        Convenience function for re-distributing global parameter values
        back into each of the original `CurveFitter.params` attributes.
        """
        for name, param in params.items():
            fitter_i, original_name = self.new_param_reference[name]
            self.original_params[fitter_i][original_name].value = param._getval()


def _parallel_residuals_calculator(params, fitfunc=None, data_tuple=None,
                                   mask=None, fcn_args=(), fcn_kws=None,
                                   model=False, costfun=None):
    """
    Objective function calculating the residuals for a curvefit. This is a
    separate function and not a method in CurveFitter to allow for
    multiprocessing.
    """
    kws = {}
    if fcn_kws is not None:
        kws = fcn_kws

    x, y, e = data_tuple

    resid = fitfunc(x, params, *fcn_args, **kws)
    if model:
        return resid

    if costfun is not None:
        return costfun(params, resid, y, e, *fcn_args, **kws)

    resid -= y
    resid /= e

    if mask is not None:
        resid_ma = ma.array(resid, mask=mask)
        return resid_ma[~resid_ma.mask].data
    else:
        return resid


def _parallel_likelihood_calculator(params, fitfunc=None, data_tuple=None,
                                    mask=None, fcn_args=(), fcn_kws=None,
                                    lnpost=None):
    """
    Function calculating the log-likelihood for a curvefit. This is a
    separate function and not a method in CurveFitter to allow for
    multiprocessing.
    """
    kws = {}
    if fcn_kws is not None:
        kws = fcn_kws

    x, y, e = data_tuple

    resid = fitfunc(x, params, *fcn_args, **kws)

    if lnpost is not None:
        return lnpost(params, resid, y, e, *fcn_args, **kws)
    else:
        resid -= y
        resid /= e
        resid *= resid

        resid += np.log(2 * np.pi * e**2)

        if mask is not None:
            resid_ma = ma.array(resid, mask=mask)
            return -0.5 * np.sum(resid_ma[~resid_ma.mask].data)
        else:
            return -0.5 * np.sum(resid)


def _parallel_global_fitfunc(x, params, fitfuncs=None,
                             new_param_reference=None, original_params=None,
                             original_userargs=None, original_kws=None):
    """
    Fit function calculating a predictive model for a curvefit. This is a
    separate function and not a method in CurveFitter to allow for
    multiprocessing.
    """
    # distribute params
    for name, param in params.items():
        fitter_i, original_name = new_param_reference[name]
        original_params[fitter_i][original_name].value = param._getval()

    model = np.zeros(0, dtype='float64')

    for i, fitfunc in enumerate(fitfuncs):
        model_i = fitfuncs[i](x[i],
                              original_params[i],
                              *original_userargs[i],
                              **original_kws[i])
        model = np.append(model,
                          model_i)
    return model


if __name__ == '__main__':
    from lmfit import fit_report

    def gauss(x, params, *args):
        """Calculates a Gaussian model"""
        p = params.valuesdict().values()
        return p[0] + p[1] * np.exp(-((x - p[2]) / p[3])**2)

    xdata = np.linspace(-4, 4, 100)
    p0 = np.array([0., 1., 0., 1.])
    bounds = [(-1., 1.), (0., 2.), (-3., 3.), (0.001, 2.)]

    temp_pars = to_parameters(p0, bounds=bounds)
    pars = to_parameters(p0 + 0.2, bounds=bounds)

    ydata = gauss(xdata, temp_pars) + 0.1 * np.random.random(xdata.size)

    f = CurveFitter(gauss, (xdata, ydata), pars)
    f.fit()

    print(fit_report(f.params))
