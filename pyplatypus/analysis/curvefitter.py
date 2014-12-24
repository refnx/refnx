# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:37:29 2014

@author: anz
"""
from lmfit import Minimizer, Parameters
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
        lowlim = np.array(bounds)[:, 0]
        hilim = np.array(bounds)[:, 1]
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
            hilim[i] += _MACHEPS
            _p0[i] = lowlim[i]
            _varies[i] = False
            warnings.warn('Parameter min==max and parameter %s was varying. %s'
                          ' now fixed' % (names[i], names[i]), RuntimeWarning)
                
        p.add(names[i], value=_p0[i], min=lowlim[i], max=hilim[i],
              vary=_varies[i], expr=expr[i])

    return p


class GlobalFitter(Minimizer):
    """
    A class for simultaneous curvefitting of multiple datasets
    """
    def __init__(self, datasets, constraints=(), minimizer_kwds=None,
                 callback=None):
        """
        datasets : sequence of CurveFitter instances
            Contains all the datasets and fitfunctions for the global fit.
        constraints : str sequence, optional
            Of the type 'dNpM:constraint'. Sets a constraint expression for
            parameter M in dataset N.  The constraint 'd2p3:d0p1' constrains
            parameter 3 in dataset 2 to be equal to parameter 1 in dataset 0.
        minimizer_kwds : dict, optional
            Extra minimization keywords to be passed to the minimizer of
            choice.
        callback : callable, optional
            Function called at each step of the minimization. Has the signature
            ``callback(params, iter, resid)``

        """
        self.callback = callback
        min_kwds = {}
        if minimizer_kwds is not None:
            min_kwds = minimizer_kwds

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
            old_names = dict((v, k) for k, v in new_names.iteritems())
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
        super(GlobalFitter, self).__init__(self.residuals,
                                           self.params,
                                           iter_cb=self.callback,
                                           **min_kwds)

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
        success = self.minimize(method=method)
        self.residuals(self.params)
        return success


class CurveFitter(Minimizer):
    """
    A curvefitting class that extends lmfit.Minimize
    """
    def __init__(self, params, xdata, ydata, fitfunc, edata=None, args=(),
                 kwds=None, minimizer_kwds=None, callback=None):
        """
        params : lmfit.Parameters instance
            Specifies the parameter set for the fit
        xdata : np.ndarray
            The independent variables
        ydata : np.ndarray
            The dependent (observed) variable
        fitfunc : callable
            Function calculating the model for the fit.  Should have the
            signature: ``fitfunc(xdata, params, *args, **kwds)``
        edata : np.ndarray, optional
            The measured uncertainty in the dependent variable, expressed as
            sd.  If this array is not specified, then edata is set to unity.
        args : tuple, optional
            Extra parameters required to fully specify fitfunc.
        kwds : dict, optional
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
        self.params = params
        if edata is not None:
            self.edata = np.asfarray(edata)
        else:
            self.edata = np.ones_like(self.ydata)

        self.args = args

        self.kwds = {}
        if kwds is not None:
            self.kwds = kwds

        min_kwds = {}
        if minimizer_kwds is not None:
            min_kwds = minimizer_kwds

        self.callback = None
        if callable(callback):
            self.callback = callback

        super(CurveFitter, self).__init__(self.residuals,
                                          self.params,
                                          iter_cb=self.callback,
                                          **min_kwds)

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
        return self.fitfunc(self.xdata, params, *self.args, **self.kwds)

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

    f = CurveFitter(pars, xdata, ydata, gauss)
    f.fit('differential_evolution')
    print fit_report(f)

    g = GlobalFitter([f], ['d0p3:1'])
    g.fit()
    print fit_report(g)
