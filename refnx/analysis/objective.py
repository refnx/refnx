from __future__ import division, print_function

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from refnx.util import ErrorProp as EP
from refnx._lib import flatten, approx_hess2
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
        """
        Returns
        -------
        varying_parameters : np.ndarray
            The parameters varying in this objective function.
        """
        return self.parameters

    def covar(self):
        """
        Returns
        -------
        covar : np.ndarray
            The covariance matrix for the fitting system
        """
        _pvals = np.array(self.varying_parameters())
        hess = approx_hess2(_pvals, self.nll)
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

    def __repr__(self):
        s = ["{:_>80}".format('')]
        s.append('Objective - {0}'.format(id(self)))

        # dataset name
        if self.data.name is None:
            s.append('Dataset = {0}'.format(repr(self.data)))
        else:
            s.append('Dataset = {0}'.format(self.data.name))

        s.append('datapoints = {0}'.format(self.npoints))
        s.append('chi2 = {0}'.format(self.chisqr()))
        s.append('Weighted = {0}'.format(self.weighted))
        s.append('Transform = {0}'.format(self.transform))
        s.append(repr(self.parameters))

        return '\n'.join(s)

    @property
    def weighted(self):
        """
        Returns
        -------
        weighted : bool
            Does the data have weights (`data.y_err`)?
        """
        return self.data.weighted

    @property
    def npoints(self):
        """
        Returns
        -------
        npoints : int
            The number of points in the dataset.
        """
        return self.data.y.size

    def varying_parameters(self):
        """
        Returns
        -------
        varying_parameters : Parameters
            A Parameters instance containing the varying Parameter objects
            that are allowed to vary during the fit.
        """
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

    def generative(self, pvals=None):
        """
        Calculate the generative function associated with the model

        Parameters
        ----------
        pvals : np.ndarray
            Parameter values for evaluation

        Returns
        -------
        model : np.ndarray
        """
        self.setp(pvals)
        return self.model(self.data.x, x_err=self.data.x_err)

    def residuals(self, pvals=None):
        """
        Calculates the residuals for a given fitting system.

        Parameters
        ----------
        pvals : np.ndarray
            Parameter values for evaluation

        Returns
        -------
        residuals : np.ndarray
            Residuals, `(data.y - model) / y_err`.
        """
        self.setp(pvals)

        model = self.model(self.data.x, x_err=self.data.x_err)
        # TODO add in varying parameter residuals? (z-scores...)

        y, y_err, model = self._data_transform(model)

        return (y - model) / y_err

    def chisqr(self, pvals=None):
        """
        Calculates the chi-squared value for a given fitting system.

        Parameters
        ----------
        pvals : np.ndarray
            Parameter values for evaluation

        Returns
        -------
        chisqr : np.ndarray
            Chi-squared value, `np.sum(residuals**2)`.
        """
        # TODO reduced chisqr? include z-scores for parameters? DOF?
        self.setp(pvals)

        return np.sum(self.residuals(pvals)**2)

    @property
    def parameters(self):
        """
        All the Parameters contained in the fitting system.

        Returns
        -------
        parameters : Parameters
            Parameters instance containing all the Parameter(s)
        """
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

        if not np.isfinite(lnprior):
            return -np.inf

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

        # if the initial attempt with unscaled values fails, than fallover
        # to situation where we scale the parameters to 1. The initial attempt
        # could fail, for example, if the data is scaled with a log10
        # transform. If the nd.MaxStepGenerator makes some points negative then
        # the transform will complain because it's trying to take the log of a
        # negative number. By scaling the pvals to 1 in the Jacobian
        # calculation we can get around this because the stepper won't make any
        # of the parameter values change sign (step too small).
        used_residuals_scaler = False

        def residuals_scaler(vals):
            return np.squeeze(self.residuals(_pvals * vals))

        try:
            with np.errstate(invalid='raise'):
                jac = approx_derivative(residuals_scaler, np.ones_like(_pvals))
            used_residuals_scaler = True
        except FloatingPointError:
            jac = approx_derivative(self.residuals, _pvals)

        if len(_pvals) == 1:
            jac = jac.T

        # need to create this because GlobalObjective does not have
        # access to all the datapoints being fitted.
        n_datapoints = np.size(jac, 0)

        # covar = J.T x J.
        # not sure why this is preferred over Hessian
        covar = np.linalg.inv(np.matmul(jac.T, jac))

        if used_residuals_scaler:
            # unwind the scaling.
            covar *= _pvals ** 2

        self.setp(_pvals)
        scale = 1.

        # scale by reduced chi2 if experimental uncertainties weren't used.
        if not (self.use_weights and self.weighted):
            scale = (self.chisqr() /
                     (n_datapoints - len(self.varying_parameters())))

        return covar * scale

    def pgen(self, n_gen=1000):
        """
        Yield random parameter vectors (only those varying) from the MCMC
        samples. The objective state is not altered.

        Parameters
        ----------
        n_gen : int, optional
            the number of samples

        Yields
        ------
        pvec : np.ndarray
            A randomly chosen parameter vector
        """
        chains = np.array([np.ravel(param.chain) for param
                           in self.varying_parameters()
                           if param.chain is not None])

        if len(chains) != len(self.varying_parameters()) or len(chains) == 0:
            raise ValueError("You need to perform sampling on all the varying"
                             "parameters first")

        samples = np.arange(np.size(chains, 1))
        choices = np.random.choice(samples, size=(n_gen,), replace=False)

        for choice in choices:
            yield chains[..., choice]


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

    def __repr__(self):
        s = ["{:_>80}".format('\n')]
        s.append('--Global Objective--')
        for obj in self.objectives:
            s.append(repr(obj))
            s.append('\n')
        return '\n'.join(s)

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

    Notes
    -----
    You ask for a transform to be carried out by calling the Transform object
    directly.

    >>> t = Transform('logY')
    >>> y, e = t(x, y, y_err)
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
        return self.__transform(x, y, y_err=y_err)

    def __transform(self, x, y, y_err=None):
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
