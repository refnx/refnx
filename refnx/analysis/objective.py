from __future__ import division, print_function

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize._numdiff import approx_derivative
import scipy.stats as stats

from refnx.util import ErrorProp as EP
from refnx._lib import flatten, approx_hess2
from refnx._lib import unique as f_unique
from refnx.dataset import Data1D
from refnx.analysis import (is_parameter, Parameter, possibly_create_parameter,
                            is_parameters, Parameters, Interval, PDF)


class BaseObjective(object):
    """Don't necessarily have to use Parameters, could use np.array"""
    def __init__(self, p, logl, logp=None, fcn_args=(), fcn_kwds=None,
                 name=None):
        self.name = name
        self.parameters = p
        self.nvary = len(p)
        self._logl = logl
        self._logp = logp
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

        return -self.logl(vals)

    def logp(self, pvals=None):
        """
        Log-prior probability function

        Parameters
        ----------
        pvals : np.ndarray
            Array containing the values to be tested.

        Returns
        -------
        logp : float
            log-prior probability

        """
        vals = self.parameters
        if pvals is not None:
            vals = pvals

        if callable(self._logp):
            return self._logp(vals, *self.fcn_args, **self.fcn_kwds)
        return 0

    def logl(self, pvals=None):
        """
        Log-likelihood probability function

        Parameters
        ----------
        pvals : np.ndarray
            Array containing the values to be tested.

        Returns
        -------
        logl : float
            log-likelihood probability.

        """
        vals = self.parameters
        if pvals is not None:
            vals = pvals

        return self._logl(vals, *self.fcn_args, **self.fcn_kwds)

    def logpost(self, pvals=None):
        """
        Log-posterior probability function

        Parameters
        ----------
        pvals : np.ndarray
            Array containing the values to be tested.

        Returns
        -------
        logpost : float
            log-probability.

        Notes
        -----
        The log probability is the sum is the sum of the log-prior and
        log-likelihood probabilities. Does not set the parameter attribute.

        """
        vals = self.parameters
        if pvals is not None:
            vals = pvals

            logpost = self.logp(vals)
        if not np.isfinite(logpost):
            return -np.inf
        logpost += self.logl(vals)
        return logpost

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
    `refnx.analysis.curvefitter.CurveFitter`.

    Parameters
    ----------
    model : refnx.analysis.Model
        the generative model function. One can also provide an object that
        inherits `refnx.analysis.Model`.
    data : refnx.dataset.Data1D
        data to be analysed.
    lnsigma : float or refnx.analysis.Parameter, optional
        Used if the  experimental uncertainty (`data.y_err`) underestimated by
        a constant fractional amount. The experimental uncertainty is modified
        as:

        `s_n**2 = y_err**2 + exp(lnsigma * 2) * model**2`

        See `Objective.logl` for more details.
    use_weights : bool
        use experimental uncertainty in calculation of residuals and
        logl, if available. If this is set to False, then you should also
        set `self.lnsigma.vary = False`, it will have no effect on the fit.
    transform : callable, optional
        the model, data and data uncertainty are transformed by this
        function before calculating the likelihood/residuals. Has the
        signature `transform(data.x, y, y_err=None)`, returning the tuple
        (`transformed_y, transformed_y_err`).
    logp_extra : callable, optional
        user specifiable log-probability term. This contribution is in
        addition to the log-prior term of the `model` parameters, and
        `model.logp`, as well as the log-likelihood of the `data`. Has
        signature:
        `logp_extra(model, data)`. The `model` will already possess
        updated parameters. Beware of including the same log-probability
        terms more than once.
    name : str
        Name for the objective.

    Notes
    -----
    For parallelisation `logp_extra` needs to be picklable.

    """

    def __init__(self, model, data, lnsigma=None, use_weights=True,
                 transform=None, logp_extra=None, name=None):
        self.model = model
        # should be a Data1D instance
        if isinstance(data, Data1D):
            self.data = data
        else:
            self.data = Data1D(data=data)

        self.lnsigma = lnsigma
        if lnsigma is not None:
            self.lnsigma = possibly_create_parameter(lnsigma, 'lnsigma')

        self.__use_weights = use_weights
        self.transform = transform
        self.logp_extra = logp_extra
        self.name = name
        if name is None:
            self.name = id(self)

    def __repr__(self):
        s = ["{:_>80}".format('')]
        s.append('Objective - {0}'.format(self.name))

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
        **bool** Does the data have weights (`data.y_err`), and is the
        objective using them?

        """
        return self.data.weighted and self.__use_weights

    @weighted.setter
    def weighted(self, use_weights):
        self.__use_weights = bool(use_weights)

    @property
    def npoints(self):
        """
        **int** the number of points in the dataset.

        """
        return self.data.y.size

    def varying_parameters(self):
        """
        Returns
        -------
        varying_parameters : refnx.analysis.Parameters
            The varying Parameter objects allowed to vary during the fit.

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

        y_err = 1.
        if self.weighted:
            y_err = self.data.y_err

        if self.transform is None:
            return y, y_err, model
        else:
            if model is not None:
                model, _ = self.transform(x, model)

            y, y_err = self.transform(x, y, y_err)
            if self.weighted:
                return y, y_err, model
            else:
                return y, 1, model

    def generative(self, pvals=None):
        """
        Calculate the generative (dependent variable) function associated with
        the model.

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

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
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

        Returns
        -------
        residuals : np.ndarray
            Residuals, `(data.y - model) / y_err`.

        """
        self.setp(pvals)

        model = self.model(self.data.x, x_err=self.data.x_err)
        # TODO add in varying parameter residuals? (z-scores...)

        y, y_err, model = self._data_transform(model)

        if self.lnsigma is not None:
            s_n = np.sqrt(y_err * y_err +
                          np.exp(2 * float(self.lnsigma)) * model * model)
        else:
            s_n = y_err

        return np.squeeze((y - model) / s_n)

    def chisqr(self, pvals=None):
        """
        Calculates the chi-squared value for a given fitting system.

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

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
        :class:`refnx.analysis.Parameters`, all the Parameters contained in the
        fitting system.

        """
        if is_parameter(self.lnsigma):
            return self.lnsigma | self.model.parameters
        else:
            return self.model.parameters

    def setp(self, pvals):
        """
        Set the parameters from pvals.

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

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

    def logp(self, pvals=None):
        """
        Calculate the log-prior of the system

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

        Returns
        -------
        logp : float
            log-prior probability

        Notes
        -----
        The log-prior is calculated as:

        .. code-block:: python

            logp = np.sum(param.logp() for param in
                             self.varying_parameters())
            logp += self.model.logp()
            logp += self.logp_extra(self.model, self.data)

        The major components of the log-prior probability are from the varying
        parameters and the Model used to construct the Objective.
        `self.model.logp` should not include any contributions from
        `self.model.parameters` otherwise they'll be counted more than once.
        The same argument applies to the user specifiable `logp_extra`
        function.

        """
        self.setp(pvals)

        logp = np.sum([param.logp() for param in self.varying_parameters()])

        if not np.isfinite(logp):
            return -np.inf

        logp += self.model.logp()

        if not np.isfinite(logp):
            return -np.inf

        if self.logp_extra is not None:
            logp += self.logp_extra(self.model, self.data)

        return logp

    def logl(self, pvals=None):
        """
        Calculate the log-likelhood of the system

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

        Returns
        -------
        logl : float
            log-likelihood probability

        Notes
        -----
        The log-likelihood is calculated as:

        .. code-block:: python

            logl = -0.5 * np.sum(((y - model) / s_n)**2
                                 + np.log(2 * pi * s_n**2))

        where

        .. code-block:: python

            s_n**2 = y_err**2 + exp(2 * lnsigma) * model**2

        """
        self.setp(pvals)

        model = self.model(self.data.x, x_err=self.data.x_err)

        logl = 0.

        y, y_err, model = self._data_transform(model)

        if self.lnsigma is not None:
            var_y = (y_err * y_err +
                     np.exp(2 * float(self.lnsigma)) * model * model)
        else:
            var_y = y_err ** 2

        # TODO do something sensible if data isn't weighted
        if self.weighted:
            logl += np.log(2 * np.pi * var_y)

        logl += (y - model)**2 / var_y

        # nans play havoc
        if np.isnan(logl).any():
            raise RuntimeError("Objective.logl encountered a NaN")

        return -0.5 * np.sum(logl)

    def nll(self, pvals=None):
        """
        Negative log-likelihood function

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

        Returns
        -------
        nll : float
            negative log-likelihood

        """
        self.setp(pvals)
        return -self.logl()

    def logpost(self, pvals=None):
        """
        Calculate the log-probability of the curvefitting system

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

        Returns
        -------
        logpost : float
            log-probability

        Notes
        -----
        The overall log-probability is the sum of the log-prior and
        log-likelihood. The log-likelihood is not calculated if the log-prior
        is impossible (`logp == -np.inf`).

        """
        self.setp(pvals)
        logpost = self.logp()

        # only calculate the probability if the parameters have finite
        # log-prior
        if not np.isfinite(logpost):
            return -np.inf

        logpost += self.logl()
        return logpost

    def covar(self):
        """
        Estimates the covariance matrix of the curvefitting system.

        Returns
        -------
        covar : np.ndarray
            Covariance matrix

        """
        _pvals = np.array(self.varying_parameters())

        used_residuals_scaler = False

        def residuals_scaler(vals):
            return np.squeeze(self.residuals(_pvals * vals))

        try:
            # we should be able to calculate a Jacobian for a parameter whose
            # value is zero. However, the scaling approach won't work.
            # This will force Jacobian calculation by unscaled parameters
            if np.any(_pvals == 0):
                raise FloatingPointError()

            with np.errstate(invalid='raise'):
                jac = approx_derivative(residuals_scaler, np.ones_like(_pvals))
            used_residuals_scaler = True
        except FloatingPointError:
            jac = approx_derivative(self.residuals, _pvals)
        finally:
            # using approx_derivative changes the state of the objective
            # parameters have to make sure they're set at the end
            self.setp(_pvals)

        # need to create this because GlobalObjective does not have
        # access to all the datapoints being fitted.
        n_datapoints = np.size(jac, 0)

        # covar = J.T x J

        # from scipy.optimize.minpack.py
        # eliminates singular parameters
        _, s, VT = np.linalg.svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        covar = np.dot(VT.T / s ** 2, VT)

        if used_residuals_scaler:
            # unwind the scaling.
            covar = covar * np.atleast_2d(_pvals) * np.atleast_2d(_pvals).T

        pvar = np.diagonal(covar).copy()
        psingular = np.where(pvar == 0)[0]

        if len(psingular) > 0:
            var_params = self.varying_parameters()
            singular_params = [var_params[ps] for ps in psingular]

            raise LinAlgError("The following Parameters have no effect on"
                              " Objective.residuals, please consider fixing"
                              " them.\n" + repr(singular_params))

        scale = 1.
        # scale by reduced chi2 if experimental uncertainties weren't used.
        if not (self.weighted):
            scale = (self.chisqr() /
                     (n_datapoints - len(self.varying_parameters())))

        return covar * scale

    def pgen(self, ngen=1000, nburn=0, nthin=1):
        """
        Yield random parameter vectors (only those varying) from the MCMC
        samples. The objective state is not altered.

        Parameters
        ----------
        ngen : int, optional
            the number of samples to yield. The actual number of samples
            yielded is `min(ngen, chain.size)`
        nburn : int, optional
            discard this many steps from the start of the chain
        nthin : int, optional
            only accept every `nthin` samples from the chain

        Yields
        ------
        pvec : np.ndarray
            A randomly chosen parameter vector

        """
        chains = np.array([np.ravel(param.chain[..., nburn::nthin]) for param
                           in self.varying_parameters()
                           if param.chain is not None])

        if len(chains) != len(self.varying_parameters()) or len(chains) == 0:
            raise ValueError("You need to perform sampling on all the varying"
                             "parameters first")

        samples = np.arange(np.size(chains, 1))

        choices = np.random.choice(samples,
                                   size=(min(ngen, samples.size),),
                                   replace=False)

        for choice in choices:
            yield chains[..., choice]

    def plot(self, pvals=None, samples=0, parameter=None):
        """
        Plot the data/model.

        Requires matplotlib be installed.

        Parameters
        ----------
        pvals : np.ndarray, optional
            Numeric values for the Parameter's that are varying
        samples: number
            If the objective has been sampled, how many samples you wish to
            plot on the graph.
        parameter: refnx.analysis.Parameter
            Creates an interactive plot for the Parameter in Jupyter. Requires
            ipywidgets be installed. Use with %matplotlib notebook/qt.

        Returns
        -------
        fig, ax : :class:`matplotlib.Figure`, :class:`matplotlib.Axes`
            `matplotlib` figure and axes objects.

        """
        import matplotlib.pyplot as plt

        self.setp(pvals)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        y, y_err, model = self._data_transform(model=self.generative())

        # add the data (in a transformed fashion)
        if self.weighted:
            ax.errorbar(self.data.x, y, y_err, color='blue',
                        label=self.data.name,
                        marker='o', ms=3, lw=0, elinewidth=2)
        else:
            ax.scatter(self.data.x, y, color='blue',
                       ms=3, label=self.data.name)

        if samples > 0:
            saved_params = np.array(self.parameters)
            # Get a number of chains, chosen randomly, set the objective,
            # and plot the model.
            for pvec in self.pgen(ngen=samples):
                y, y_err, model = self._data_transform(
                    model=self.generative(pvec))

                ax.plot(self.data.x,
                        model,
                        color="k", alpha=0.01)

            # put back saved_params
            self.setp(saved_params)

        # add the fit
        generative_plot = ax.plot(self.data.x, model, color='red', zorder=20)

        if parameter is None:
            return fig, ax

        # create an interactive plot in a Jupyter notebook.
        def f(val):
            if parameter is not None:
                parameter.value = float(val)
            y, y_err, model = self._data_transform(model=self.generative())
            generative_plot[0].set_data(self.data.x, model)
            fig.canvas.draw()

        import ipywidgets
        return fig, ax, ipywidgets.interact(f, val=float(parameter))

    def corner(self, **kwds):
        """
        Corner plot of the chains belonging to the Parameters.
        Requires the `corner` and `matplotlib` packages.

        Parameters
        ----------
        kwds: dict
            passed directly to the `corner.corner` function

        Returns
        -------
        fig : :class:`matplotlib.Figure` object.
        """
        import corner
        var_pars = self.varying_parameters()
        chain = np.array([par.chain for par in var_pars])
        labels = [par.name for par in var_pars]
        chain = chain.reshape(len(chain), -1).T
        return corner.corner(chain, labels=labels, **kwds)


class GlobalObjective(Objective):
    """
    Global Objective function for simultaneous fitting with
    `refnx.analysis.CurveFitter`

    Parameters
    ----------
    objectives : list
        list of :class:`refnx.analysis.Objective` objects

    """

    def __init__(self, objectives):
        self.objectives = objectives
        weighted = [objective.weighted for objective in objectives]

        self._weighted = np.array(weighted, dtype=bool)

        if len(np.unique(self._weighted)) > 1:
            raise ValueError("All the objectives must be either weighted or"
                             " unweighted, you cannot have a mixture.")

    def __repr__(self):
        s = ["{:_>80}".format('\n')]
        s.append('--Global Objective--')
        for obj in self.objectives:
            s.append(repr(obj))
            s.append('\n')
        return '\n'.join(s)

    @property
    def weighted(self):
        """
        **bool** do all the datasets have y_err, and are all the objectives
        wanting to use weights?

        """
        return self._weighted.all()

    @property
    def npoints(self):
        """
        **int** number of data points in all the objectives.

        """
        npoints = 0
        for objective in self.objectives:
            npoints += objective.npoints
        return npoints

    def residuals(self, pvals=None):
        """
        Concatenated residuals for each of the
        :meth:`refnx.analysis.Objective.residuals`.

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

        Returns
        -------
        residuals : np.ndarray
            Concatenated :meth:`refnx.analysis.Objective.residuals`

        """
        self.setp(pvals)

        residuals = []
        for objective in self.objectives:
            residual = objective.residuals()
            residuals.append(residual)

        return np.concatenate(residuals)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with all the objectives.

        """
        # TODO this is probably going to be slow.
        # cache and update strategy?
        p = Parameters(name='global fitting parameters')

        for objective in self.objectives:
            p.append(objective.parameters)

        return p

    def logp(self, pvals=None):
        """
        Calculate the log-prior of the system

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters, optional
            values for the varying or entire set of parameters

        Returns
        -------
        logp : float
            log-prior probability

        """
        self.setp(pvals)

        logp = 0.
        for objective in self.objectives:
            logp += objective.logp()
            # shortcut if one of the priors is impossible
            if not np.isfinite(logp):
                return -np.inf

        return logp

    def logl(self, pvals=None):
        """
        Calculate the log-likelhood of the system

        Parameters
        ----------
        pvals : array-like or refnx.analysis.Parameters
            values for the varying or entire set of parameters

        Returns
        -------
        logl : float
            log-likelihood probability

        """
        self.setp(pvals)
        logl = 0.

        for objective in self.objectives:
            logl += objective.logl()

        return logl

    def plot(self, parameter=None):
        """
        Plot the data/model for all the objectives in the GlobalObjective.
        Matplotlib must be installed to use this method.

        Parameters
        ----------
        parameter: refnx.analysis.Parameter
            Creates an interactive plot for the Parameter in Jupyter. Requires
            ipywidgets be installed. Use with %matplotlib notebook/qt.

        Returns
        -------
        fig, ax : :class:`matplotlib.Figure`, :class:`matplotlib.Axes`
            `matplotlib` figure and axes objects.

        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        generative_plots = []

        for objective in self.objectives:
            y, y_err, model = objective._data_transform(
                model=objective.generative())

            # add the data (in a transformed fashion)
            if objective.weighted:
                ax.errorbar(objective.data.x, y, y_err,
                            label=objective.data.name,
                            ms=3, lw=0, elinewidth=2, marker='o')
            else:
                ax.scatter(objective.data.x, y,
                           label=objective.data.name)

            # add the fit
            generative_plots.append(
                ax.plot(objective.data.x, model, color='r',
                        lw=1.5, zorder=20)[0])

        if parameter is None:
            return fig, ax

        # create an interactive plot in a Jupyter notebook.
        def f(val):
            if parameter is not None:
                parameter.value = float(val)
            for i, objective in enumerate(self.objectives):
                y, y_err, model = objective._data_transform(
                    model=objective.generative())

                generative_plots[i].set_data(objective.data.x, model)
            fig.canvas.draw()

        import ipywidgets
        return fig, ax, ipywidgets.interact(f, val=float(parameter))

        return fig, ax


class Transform(object):
    r"""
    Mathematical transforms of numeric data.

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

    >>> x = np.linspace(0.01, 0.1, 11)
    >>> y = np.linspace(100, 1000, 11)
    >>> y_err = np.sqrt(y)
    >>> t = Transform('logY')
    >>> ty, te = t(x, y, y_err)
    >>> ty
    array([2.        , 2.2787536 , 2.44715803, 2.56820172, 2.66275783,
           2.74036269, 2.80617997, 2.86332286, 2.91381385, 2.95904139,
           3.        ])

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
        """
        Calculate the transformed data

        Parameters
        ----------
        x : array-like
            x-values
        y : array-like
            y-values
        y_err : array-like
            Uncertainties in `y` (standard deviation)

        Returns
        -------
        yt, et : tuple
            The transformed data

        Examples
        --------
        >>> x = np.linspace(0.01, 0.1, 11)
        >>> y = np.linspace(100, 1000, 11)
        >>> y_err = np.sqrt(y)
        >>> t = Transform('logY')
        >>> ty, te = t(x, y, y_err)
        >>> ty
        array([2.        , 2.2787536 , 2.44715803, 2.56820172, 2.66275783,
               2.74036269, 2.80617997, 2.86332286, 2.91381385, 2.95904139,
               3.        ])

        """
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


class _pymc_objective_wrapper(object):
    """
    Wraps objective for pymc3

    Parameters
    ----------
    objective: refnx.analysis.Objective

    Notes
    -----
    This wrapper is required so that __call__ can receive a list of
    pymc3 parameters and arrange them in a suitable form to dispatch to
    the fitfunction.
    """

    def __init__(self, objective):
        self.objective = objective
        self.func = objective.model.model
        # get as close to using fitfunc as possible
        if objective.model.fitfunc is not None:
            self.func = objective.model.fitfunc

    def __name__(self):
        return 'objective'

    def __call__(self, *args):
        vals = self.func(self.objective.data.x,
                         np.array(args),
                         x_err=self.objective.data.x_err)
        return vals


def pymc_objective(objective):
    """
    Creates a pymc3 model from an Objective. Will not be able to use the NUTS
    sampler because gradients aren't evaluable.

    Requires theano and pymc3 be installed. This is an experimental feature.

    Parameters
    ----------
    objective: refnx.analysis.Objective

    Returns
    -------
    model: pymc3.Model

    Notes
    -----
    The varying parameters are renamed 'p0', 'p1', etc, as it's vital in pymc3
    that all parameters have their own unique name.

    """
    import pymc3 as pm
    import theano.tensor as T
    from theano.compile.ops import as_op

    basic_model = pm.Model()

    wrapped_obj = _pymc_objective_wrapper(objective)

    pars = objective.varying_parameters()
    wrapped_pars = []
    with basic_model:
        # Priors for unknown model parameters
        for i, par in enumerate(pars):
            name = 'p%d' % i
            p = _to_pymc3_distribution(name, par)
            wrapped_pars.append(p)

        # Expected value of outcome
        try:
            v = wrapped_obj(*wrapped_pars)
        except Exception:
            print("Falling back, theano autodiff won't work on function"
                  " object")
            o = as_op(itypes=[T.dscalar] * len(pars),
                      otypes=[T.dvector])(wrapped_obj)
            v = o(*wrapped_pars)

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal('Y_obs', mu=v, sd=objective.data.y_err,
                          observed=objective.data.y)

        if not y_obs:
            return None

    return basic_model


def _to_pymc3_distribution(name, par):
    """
    Create a pymc3 continuous distribution from a Bounds object.

    Parameters
    ----------
    name : str
        Name of parameter
    par : refnx.analysis.Parameter
        The parameter to wrap

    Returns
    -------
    d : pymc3.Distribution
        The pymc3 distribution

    """
    import pymc3 as pm
    import theano.tensor as T
    from theano.compile.ops import as_op

    dist = par.bounds
    # interval and both lb, ub are finite
    if (isinstance(dist, Interval) and
            np.isfinite([dist.lb, dist.ub]).all()):
        return pm.Uniform(name, dist.lb, dist.ub)
    # no bounds
    elif (isinstance(dist, Interval) and
          np.isneginf(dist.lb) and
          np.isinf(dist.lb)):
        return pm.Flat(name)
    # half open uniform
    elif isinstance(dist, Interval) and not np.isfinite(dist.lb):
        return dist.ub - pm.HalfFlat(name)
    # half open uniform
    elif isinstance(dist, Interval) and not np.isfinite(dist.ub):
        return dist.lb + pm.HalfFlat(name)

    # it's a PDF
    if isinstance(dist, PDF):
        dist_gen = getattr(dist.rv, 'dist', None)

        if isinstance(dist.rv, stats.rv_continuous):
            dist_gen = dist.rv

        if isinstance(dist_gen, type(stats.uniform)):
            if hasattr(dist.rv, 'args'):
                p = pm.Uniform(name, dist.rv.args[0],
                               dist.rv.args[1] + dist.rv.args[0])
            else:
                p = pm.Uniform(name, 0, 1)
            return p

        # norm from scipy.stats
        if isinstance(dist_gen, type(stats.norm)):
            if hasattr(dist.rv, 'args'):
                p = pm.Normal(name, mu=dist.rv.args[0], sd=dist.rv.args[1])
            else:
                p = pm.Normal(name, mu=0, sd=1)
            return p

    # not open, uniform, or normal, so fall back to DensityDist.
    d = as_op(itypes=[T.dscalar], otypes=[T.dscalar])(dist.logp)
    r = as_op(itypes=[T.dscalar], otypes=[T.dscalar])(dist.rvs)
    p = pm.DensityDist(name, d, random=r)

    return p
