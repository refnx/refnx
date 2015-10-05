# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:37:29 2014

@author: anz
"""
from __future__ import print_function
from lmfit import Minimizer, Parameters
import warnings
import numpy as np
import numpy.ma as ma
import re
from functools import partial
import abc

# check for EMCEE
HAS_EMCEE = False
try:
    import emcee as emcee
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


class FitFunction(object):
    """
    An abstract FitFunction class.
    """
    def __init__(self):
        pass

    def __call__(self, x, params, *args, **kws):
        return self.model(x, params, *args, **kws)

    @abc.abstractmethod
    def model(self, x, params, *args, **kws):
        """
        Override this method in your own fitfunction
        """
        raise RuntimeError("You can't use the abstract base FitFunction in a"
                           " real fit")

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


class CurveFitter(Minimizer):
    """
    A curvefitting class that extends lmfit.Minimize
    """
    def __init__(self, fitfunc, xdata, ydata, params, edata=None, mask=None,
                 fcn_args=(), fcn_kws=None, kws=None, callback=None):
        """
        fitfunc : callable
            Function calculating the generative model for the fit.  Should have
            the signature: ``fitfunc(x, params, *fcn_args, **fcn_kws)``.
        x : np.ndarray
            The independent variables
        y : np.ndarray
            The dependent (observed) variable
        params : lmfit.Parameters instance
            Specifies the parameter set for the fit
        edata : np.ndarray, optional
            The measured uncertainty in the dependent variable, expressed as
            sd.  If this array is not specified, then edata is set to unity.
        mask : np.ndarray, optional
            A boolean array with the same shape as y.  If mask is True
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
            self.mask = None

        if edata is not None:
            self.edata = np.asfarray(edata)
            self.scale_covar = False
        else:
            self.edata = np.ones_like(self.ydata)
            self.scale_covar = True

        min_kwds = {}
        if kws is not None:
            min_kwds = kws

        self._resid = partial(_parallel_residuals_calculator,
                              fitfunc=fitfunc,
                              data_tuple=(self.xdata,
                                          self.ydata,
                                          self.edata),
                              mask=mask,
                              fcn_args=fcn_args,
                              fcn_kws=fcn_kws)

        super(CurveFitter, self).__init__(self._resid,
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

    def residuals(self, params):
        """
        Calculate the difference between the data and the model. Also known as
        the objective function. This is a convenience method. Over-riding it
        will not change a fit

        residuals = (fitfunc - y) / edata

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
        return self._resid(params)

    def model(self, params):
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


class GlobalFitter(CurveFitter):
    """
    Simultaneous curvefitting of multiple datasets
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
            **Important** For a parameter (`d2:scale` in this example) to be
            constrained by this mechanism it must not have any pre-existing
            constraints within its individual fitter. If there are
            pre-existing constraints then those are honoured, and constraints
            specified here are ignored.
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
                                           xdata,
                                           np.hstack(ydata),
                                           self.params,
                                           edata=np.hstack(edata),
                                           callback=callback,
                                           kws=min_kwds)

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
        return self._fitfunc(self.xdata, params)

    def residuals(self, params):
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
        self.residuals(params)


def _parallel_residuals_calculator(params, fitfunc=None, data_tuple=None,
                                   mask=None, fcn_args=(), fcn_kws=None):
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
    resid -= y
    resid /= e

    if mask is not None:
        resid_ma = ma.array(resid, mask=mask)
        return resid_ma[~resid_ma.mask].data
    else:
        return resid


def _parallel_global_fitfunc(x, params, fitfuncs=None,
                             new_param_reference=None, original_params=None,
                             original_userargs=None, original_kws=None):
    """
    Objective function calculating the residuals for a curvefit. This is a
    separate function and not a method in CurveFitter to allow for
    multiprocessing.
    """
    # distribute params
    for name, param in params.items():
        fitter_i, original_name = new_param_reference[name]
        original_params[i][original_name].value = param._getval()

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

    temp_pars = to_Parameters(p0, bounds=bounds)
    pars = to_Parameters(p0 + 0.2, bounds=bounds)

    ydata = gauss(xdata, temp_pars) + 0.1 * np.random.random(xdata.size)

    f = CurveFitter(gauss, xdata, ydata, pars)
    f.fit()

    print(fit_report(f.params))
