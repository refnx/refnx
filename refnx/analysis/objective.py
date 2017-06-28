from __future__ import division, print_function

import numpy as np
import numdifftools as nd

from refnx.util import ErrorProp as EP
from refnx._lib import flatten
from refnx._lib import unique as f_unique
from refnx.dataset import Data1D
from refnx.analysis import (is_parameter, Parameter, possibly_create_parameter,
                            is_parameters, Parameters)


class BaseObjective(object):
    """Don't necessarily have to use Parameters, could use np.array"""
    def __init__(self, p, lnlike, lnprior=None, fcn_args=(), fcn_kwds=None):
        self.parameters = p
        self.nvary = len(p)
        self._lnlike = lnlike
        self._lnprior = lnprior
        self.fcn_args = fcn_args
        self.fcn_kwds = {}
        if fcn_kwds is not None:
            self.fcn_kwds = fcn_kwds

    def setp(self, pvals):
        """
        Set the parameters from pvals

        Parameters
        ----------
        pvals : np.ndarray
            Array containing the values to be tested.
        """
        self.parameters[:] = pvals

    def nll(self, pvals=None):
        """
        Negative log-likelihood function

        Parameters
        ----------
        pvals : np.ndarray
            Array containing the values to be tested.

        Returns
        -------
        nll : float
            negative log-likelihood
        """
        vals = self.parameters
        if pvals is not None:
            vals = pvals

        return -self.lnlike(vals)

    def lnprior(self, pvals=None):
        """
        Log-prior probability function

        Parameters
        ----------
        pvals : np.ndarray
            Array containing the values to be tested.

        Returns
        -------
        lnprior : float
            log-prior probability
        """
        vals = self.parameters
        if pvals is not None:
            vals = pvals

        if callable(self._lnprior):
            return self._lnprior(vals, *self.fcn_args, **self.fcn_kwds)
        return 0

    def lnlike(self, pvals=None):
        """
        Log-likelihood probability function

        Parameters
        ----------
        pvals : np.ndarray
            Array containing the values to be tested.

        Returns
        -------
        lnlike : float
            log-likelihood probability.
        """
        vals = self.parameters
        if pvals is not None:
            vals = pvals

        return self._lnlike(vals, *self.fcn_args, **self.fcn_kwds)

    def lnprob(self, pvals=None):
        """
        Log-probability function

        Parameters
        ----------
        pvals : np.ndarray
            Array containing the values to be tested.

        Returns
        -------
        lnprob : float
            log-probability.

        Notes
        -----
        The log probability is the sum is the sum of the log-prior and
        log-likelihood probabilities. Does not set the parameter attribute.
        """
        vals = self.parameters
        if pvals is not None:
            vals = pvals

        lnprob = self.lnprior(vals)
        if not np.isfinite(lnprob):
            return -np.inf
        lnprob += self.lnlike(vals)
        return lnprob

    def varying_parameters(self):
        return self.parameters

    def covar(self):
        _pvals = np.array(self.varying_parameters())

        step = nd.MaxStepGenerator(base_step=None, scale=3)
        hess = nd.Hessian(self.nll, step_ratio=None,
                          step=step)(_pvals)

        covar = np.linalg.inv(hess)
        self.setp(_pvals)
        return covar


class Objective(BaseObjective):
    """
    Objective function for using with curvefitters such as
    `playtime.curvefitter.CurveFitter`
    """

    def __init__(self, model, data, lnsigma=0, use_weights=True,
                 transform=None):
        """
        Parameters
        ----------
        model : playtime.model.Model
            the generative model function. One can also provide an object that
            inherits `playtime.model.Model`.
        data : refnx.dataset.Data1D
            data to be analysed.
        lnsigma : float or Parameter, optional
            the experimental uncertainty (`data.y_err`) is multiplied by
            `exp(lnsigma)`. Used when the experimental uncertainty is
            underestimated.
        use_weights : bool
            use experimental uncertainty in calculation of residuals and
            lnlike, if available. If this is set to False, then you should also
            set `self.lnsigma.vary = False`, it will have no effect on the fit.
        transform : callable, optional
            the model, data and data uncertainty are transformed by this
            function before calculating the likelihood/residuals. Has the
            signature `transform(data.x, y, y_err=None)`, returning the tuple
            `transformed_y, transformed_y_err`.
        """
        # lnsigma is a parameter for underestimated errors
        self.model = model
        # should be a Data1D instance
        if isinstance(data, Data1D):
            self.data = data
        else:
            self.data = Data1D(data=data)

        if is_parameter(lnsigma):
            self.lnsigma = lnsigma
        else:
            self.lnsigma = lnsigma

        self.use_weights = use_weights
        self.transform = transform

    @property
    def weighted(self):
        return self.data.weighted

    @property
    def npoints(self):
        return self.data.y.size

    def varying_parameters(self):
        # create and return a Parameters object because it has the
        # __array__ method, which allows one to quickly get numerical values.
        p = Parameters()
        p.data = [param for param in f_unique(flatten(self.parameters))
                  if param.vary]
        return p

    def _data_transform(self, model=None):
        x = self.data.x
        y = self.data.y
        weight = self.use_weights and self.data.weighted

        if weight:
            y_err = self.data.y_err * np.exp(float(self.lnsigma))
        else:
            y_err = 1.

        if self.transform is None:
            return y, y_err, model
        else:
            if model is not None:
                model, _ = self.transform(x, model)
                y, y_err = self.transform(x, y, y_err)
                if weight:
                    return y, y_err, model
                else:
                    return y, 1, model

    def residuals(self, pvals=None):
        self.setp(pvals)

        model = self.model(self.data.x, x_err=self.data.x_err)
        # TODO add in varying parameter residuals? (z-scores...)

        y, y_err, model = self._data_transform(model)

        return (y - model) / y_err

    def chisqr(self, pvals=None):
        # TODO reduced chisqr? include z-scores for parameters? DOF?
        self.setp(pvals)

        return np.sum(self.residuals(pvals)**2)

    @property
    def parameters(self):
        if is_parameter(self.lnsigma):
            return self.lnsigma | self.model.parameters
        else:
            return self.model.parameters

    def setp(self, pvals):
        """
        Set the parameters from pvals

        Parameters
        ----------
        pvals : np.ndarray
            Array containing the values to be tested.
        """
        if pvals is None:
            return

        # set here rather than delegating to a Parameters
        # object, because it may not necessarily be a
        # Parameters object
        _varying_parameters = self.varying_parameters()
        if len(pvals) == len(_varying_parameters):
            for idx, param in enumerate(_varying_parameters):
                param.value = pvals[idx]
            return

        # values supplied are enough to specify all parameter values
        # even those that are repeated
        flattened_parameters = list(flatten(self.parameters))
        if len(pvals) == len(flattened_parameters):
            for idx, param in enumerate(flattened_parameters):
                param.value = pvals[idx]
            return

        raise ValueError('Incorrect number of values supplied, supply either'
                         ' the full number of parameters, or only the varying'
                         ' parameters.')

    def lnprior(self, pvals=None):
        """
        Calculate the log-prior of the system

        Parameters
        ----------
        pvals : np.ndarray, optional
            Numeric values for the Parameter's that are varying

        Returns
        -------
        lnprior : float
            log-prior probability

        Notes
        -----
        The model attribute can also add extra terms to the log-prior if
        needed, but it should not include any contributions from
        `Objective.parameters` otherwise they'll be counted twice.
        """
        self.setp(pvals)

        lnprior = np.sum(param.lnprob() for param in self.varying_parameters())
        lnprior += self.model.lnprob()

        return lnprior

    def lnlike(self, pvals=None):
        """
        Calculate the log-likelhood of the system

        Parameters
        ----------
        pvals : np.ndarray, optional
            Numeric values for the Parameter's that are varying

        Returns
        -------
        lnlike : float
            log-likelihood probability
        """
        self.setp(pvals)

        model = self.model(self.data.x, x_err=self.data.x_err)

        lnlike = 0.

        y, y_err, model = self._data_transform(model)

        # TODO do something sensible if data isn't weighted
        if self.use_weights and self.data.weighted:
            # ignoring 2 * pi constant
            lnlike += np.log(y_err**2)

        lnlike += ((y - model) / y_err)**2

        return -0.5 * np.sum(lnlike)

    def nll(self, pvals=None):
        """
        Negative log-likelihood function

        Parameters
        ----------
        pvals : np.ndarray
            Array containing the values to be tested.

        Returns
        -------
        nll : float
            negative log-likelihood
        """
        self.setp(pvals)
        return -self.lnlike()

    def lnprob(self, pvals=None):
        """
        Calculate the log-probability of the curvefitting system

        Parameters
        ----------
        pvals : np.ndarray, optional
            Numeric values for the Parameter's that are varying

        Returns
        -------
        lnprob : float
            log-probability

        Notes
        -----
        The overall log-probability is the sum of the log-prior and
        log-likelihood. The log-likelihood is not calculated if the log-prior
        is impossible (`lnprior == -np.inf`).
        """
        self.setp(pvals)
        lnprob = self.lnprior()

        # only calculate the probability if the parameters have finite
        # log-prior
        if not np.isfinite(lnprob):
            return -np.inf

        lnprob += self.lnlike()
        return lnprob

    def covar(self):
        """
        Estimates the covariance matrix of the curvefitting system.
        """
        _pvals = np.array(self.varying_parameters())

        # hess = nd.Hessian(self.chisqr)(_pvals)
        step = nd.MaxStepGenerator(base_step=None, scale=3)

        # # scale values to unity
        # def residuals_scaler(vals):
        #     return self.residuals(_pvals * vals)

        jac = nd.Jacobian(self.residuals,
                          step=step)(_pvals)
        # jac = nd.Jacobian(residuals_scaler,
        #                   step=step)(np.ones_like(_pvals))

        if len(_pvals) == 1:
            jac = jac.T

        # need to create this because GlobalObjective does not have
        # access to all the datapoints being fitted.
        n_datapoints = np.size(jac, 0)

        # covar = J.T x J.
        # not sure why this is preferred over Hessian
        covar = np.linalg.inv(np.matmul(jac.T, jac))
        # unwind the scaling.
        # covar *= _pvals ** 2

        self.setp(_pvals)
        scale = 1.

        # scale by reduced chi2 if experimental uncertainties weren't used.
        if not (self.use_weights and self.weighted):
            scale = (self.chisqr() /
                     (n_datapoints - len(self.varying_parameters())))

        return covar * scale


class GlobalObjective(Objective):
    """
    Global Objective function for simultaneous fitting with
    `playtime.curvefitter.CurveFitter`
    """

    def __init__(self, objectives):
        """
        Parameters
        ----------
        objectives : list of Objective objects
        """
        self.objectives = objectives
        weighted = []
        use_weights = []
        for objective in objectives:
            weighted.append(objective.data.weighted)
            use_weights.append(objective.use_weights)
        self._weighted = np.array(weighted, dtype=bool).all()
        self._use_weights = np.array(use_weights, dtype=bool).any()
        if self._use_weights and not self._weighted:
            raise ValueError("One of the GlobalObjective.objectives wants to "
                             "use_weights, but not all the individual "
                             "objectives supplied weights")

    @property
    def use_weights(self):
        return self._use_weights

    @property
    def weighted(self):
        return self._weighted

    @property
    def npoints(self):
        npoints = 0
        for objective in self.objectives:
            npoints += objective.npoints
        return npoints

    def residuals(self, pvals=None):
        # residuals in a globalfitting context are just the individual
        # objective.residuals concatenated.
        self.setp(pvals)

        residuals = []
        for objective in self.objectives:
            residual = objective.residuals()
            residuals.append(residual)

        return np.concatenate(residuals)

    @property
    def parameters(self):
        # TODO this is probably going to be slow.
        # cache and update strategy?
        p = Parameters(name='global fitting parameters')

        for objective in self.objectives:
            p.append(objective.parameters)

        return p

    def lnprior(self, pvals=None):
        """
        Calculate the log-prior of the system

        Parameters
        ----------
        pvals : np.ndarray, optional
            Numeric values for the Parameter's that are varying

        Returns
        -------
        lnprior : float
            log-prior probability

        Notes
        -----
        The model attribute can also add extra terms to the log-prior if
        needed, but it should not include any contributions from
        `Objective.parameters` otherwise they'll be counted twice.
        """
        self.setp(pvals)

        lnprior = 0.
        for objective in self.objectives:
            lnprior += objective.lnprior()
            lnprior += objective.model.lnprob()

        return lnprior

    def lnlike(self, pvals=None):
        """
        Calculate the log-likelhood of the system

        Parameters
        ----------
        pvals : np.ndarray, optional
            Numeric values for the Parameter's that are varying

        Returns
        -------
        lnlike : float
            log-likelihood probability
        """
        self.setp(pvals)
        lnlike = 0.

        for objective in self.objectives:
            lnlike += objective.lnlike()

        return lnlike


class Transform(object):
    r"""
    Mathematical transforms of numeric data

    Parameters
    ----------
    form : None or str
        One of:

            - 'lin'
                No transform is made
            - 'logY'
                log10 transform
            - 'YX4'
                YX**4 transform
            - 'YX2'
                YX**2 transform
            - None
                No transform is made
    """
    def __init__(self, form):
        types = [None, 'lin', 'logY', 'YX4', 'YX2']
        self.form = None

        if form in types:
            self.form = form
        else:
            raise ValueError("The form parameter must be one of [None, 'lin',"
                             " 'logY', 'YX4', 'YX2']")

    def __call__(self, x, y, y_err=None):
        return self.transform(x, y, y_err=y_err)

    def transform(self, x, y, y_err=None):
        r"""
        Transform the data passed in

        Parameters
        ----------
        x : array-like

        y : array-like

        y_err : array-like

        Returns
        -------
        yt, et : tuple
            The transformed data
        """

        if y_err is None:
            etemp = np.ones_like(y)
        else:
            etemp = y_err

        if self.form in ['lin', None]:
            yt = np.copy(y)
            et = np.copy(etemp)
        elif self.form == 'logY':
            yt, et = EP.EPlog10(y, etemp)
        elif self.form == 'YX4':
            yt = y * np.power(x, 4)
            et = etemp * np.power(x, 4)
        elif self.form == 'YX2':
            yt = y * np.power(x, 2)
            et = etemp * np.power(x, 2)
        if y_err is None:
            return yt, None
        else:
            return yt, et
