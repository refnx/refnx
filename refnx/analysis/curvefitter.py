from __future__ import division

from collections import namedtuple
import sys
import re
import warnings
import array

import numpy as np

from scipy._lib._util import check_random_state
from scipy.optimize import minimize, differential_evolution, least_squares

from refnx.analysis import Objective, Interval, PDF, is_parameter
from refnx._lib import (unique as f_unique, possibly_create_pool,
                        possibly_open_file, flatten)
from refnx._lib.util import getargspec

from refnx._lib import emcee

# PTSampler has been forked into a separate package. Try both places
_HAVE_PTSAMPLER = False
PTSampler = type(None)

try:
    from ptemcee.sampler import Sampler as PTSampler
    _HAVE_PTSAMPLER = True
except ImportError:
    warnings.warn("PTSampler (parallel tempering) is not available,"
                  " please install the ptemcee package", ImportWarning)

MCMCResult = namedtuple('MCMCResult', ['name', 'param', 'stderr', 'chain',
                                       'median'])


def _objective_lnprob(theta, userargs=()):
    """
    Calculates the log-posterior probability.

    Parameters
    ----------
    theta : sequence
        Float parameter values (only those being varied)
    userargs : tuple, optional
        Extra positional arguments required for user objective function

    Returns
    -------
    lnprob : float
        Log posterior probability

    """
    # need to use this function because PY27 can't pickle a partial on
    # an object method
    objective = userargs
    return objective.lnprob(theta)


def _objective_lnlike(theta, userargs=()):
    """
    Calculates the log-likelihood probability.

    Parameters
    ----------
    theta : sequence
        Float parameter values (only those being varied)
    userargs : tuple, optional
        Extra positional arguments required for user objective function

    Returns
    -------
    lnlike : float
        Log likelihood probability

    """
    # need to use this function because PY27 can't pickle a partial on
    # an object method
    objective = userargs
    return objective.lnlike(theta)


def _objective_lnprior(theta, userargs=()):
    """
    Calculates the log-prior probability.

    Parameters
    ----------
    theta : sequence
        Float parameter values (only those being varied)
    userargs : tuple, optional
        Extra positional arguments required for user objective function

    Returns
    -------
    lnprior : float
        Log prior probability

    """
    # need to use this function because PY27 can't pickle a partial on
    # an object method
    objective = userargs
    return objective.lnprior(theta)


class CurveFitter(object):
    """
    Analyse a curvefitting system (with MCMC sampling)

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The :class:`refnx.analysis.Objective` to be analysed.
    nwalkers : int, optional
        How many walkers you would like the sampler to have. Must be an
        even number. The more walkers the better.
    ntemps : int or None, optional
        If `ntemps == -1`, then an :class:`emcee.EnsembleSampler` is used
        during the `sample` method.
        Otherwise, or if `ntemps is None` then parallel tempering is
        used with a :class:`ptemcee.sampler.Sampler` object during the `sample`
        method, with `ntemps` specifing the number of temperatures. Can be
        `None`, in which case the `Tmax` keyword argument sets the maximum
        temperature. Parallel Tempering is useful if you expect your
        posterior distribution to be multi-modal.

    mcmc_kws : dict
        Keywords used to create the :class:`emcee.EnsembleSampler` or
        :class:`ptemcee.sampler.Sampler` objects.

    Notes
    -----
    See the documentation at http://dan.iel.fm/emcee/current/api/ for
    further details on what keywords are permitted, and for further
    information on Parallel Tempering. The `pool` and `threads` keywords
    are ignored here. Specification of parallel threading is done with the
    `pool` argument in the `sample` method.
    """
    def __init__(self, objective, nwalkers=200, ntemps=-1, **mcmc_kws):
        """
        Parameters
        ----------
        objective : refnx.analysis.Objective
            The :class:`refnx.analysis.Objective` to be analysed.
        nwalkers : int, optional
            How many walkers you would like the sampler to have. Must be an
            even number. The more walkers the better.
        ntemps : int or None, optional
            If `ntemps == -1`, then an :class:`emcee.EnsembleSampler` is used
            during the `sample` method.
            Otherwise, or if `ntemps is None` then parallel tempering is
            used with a :class:`ptemcee.sampler.Sampler` object during the
            `sample` method, with `ntemps` specifing the number of
            temperatures. Can be `None`, in which case the `Tmax` keyword
            argument sets the maximum temperature. Parallel Tempering is
            useful if you expect your posterior distribution to be multi-modal.
        mcmc_kws : dict
            Keywords used to create the :class:`emcee.EnsembleSampler` or
            :class:`ptemcee.sampler.PTSampler` objects.

        Notes
        -----
        See the documentation at http://dan.iel.fm/emcee/current/api/ for
        further details on what keywords are permitted. The `pool` and
        keyword is ignored here. Specification of parallel threading is done
        with the `pool` argument in the `sample` method.
        To use parallel tempering you will need to install the
        :package:`ptemcee` package.
        """
        self.objective = objective
        self._varying_parameters = objective.varying_parameters()
        self.nvary = len(self._varying_parameters)
        if not self.nvary:
            raise ValueError("No parameters are being fitted")

        self.mcmc_kws = {}
        if mcmc_kws is not None:
            self.mcmc_kws.update(mcmc_kws)

        if 'pool' in self.mcmc_kws:
            self.mcmc_kws.pop('pool')
        if 'threads' in self.mcmc_kws:
            self.mcmc_kws.pop('threads')

        self._nwalkers = nwalkers
        self._ntemps = ntemps

        if ntemps == -1:
            self.mcmc_kws['args'] = (objective,)
            self.sampler = emcee.EnsembleSampler(nwalkers,
                                                 self.nvary,
                                                 _objective_lnprob,
                                                 **self.mcmc_kws)
        # Parallel Tempering was requested.
        else:
            if not _HAVE_PTSAMPLER:
                raise RuntimeError("You need to install the 'ptemcee' package"
                                   " to use parallel tempering")

            sig = {'loglargs': (objective,),
                   'logpargs': (objective,),
                   'ntemps': ntemps,
                   'nwalkers': nwalkers,
                   'dim': self.nvary,
                   'logl': _objective_lnlike,
                   'logp': _objective_lnprior
                   }
            sig.update(self.mcmc_kws)
            self.sampler = PTSampler(**sig)

            # construction of the PTSampler creates an ntemps attribute.
            # If it was constructed with ntemps = None, then ntemps will
            # be an integer.
            self._ntemps = self.sampler.ntemps

        self._lastpos = None

    def initialise(self, pos='covar'):
        """
        Initialise the emcee walkers.

        Parameters
        ----------
        pos : str or np.ndarray
            Method for initialising the emcee walkers. One of:

            - 'covar', use the estimated covariance of the system.
            - 'jitter', add a small amount of gaussian noise to each parameter
            - 'prior', sample random locations from the prior
            - pos, an array that specifies a snapshot of the walkers. Has shape
                `(nwalkers, ndim)`, or `(ntemps, nwalkers, ndim)` if parallel
                 tempering is employed. You can also provide a previously
                 created chain.
        """
        nwalkers = self._nwalkers
        nvary = self.nvary

        # account for parallel tempering
        _ntemps = self._ntemps

        # If you're not doing parallel tempering, temporarily set the number of
        # temperatures to be created to 1, thereby producing initial positions
        # of (1, nwalkers, nvary), this first dimension should be removed at
        # the end of the method
        if self._ntemps == -1:
            _ntemps = 1

        # position is specified with array (no parallel tempering)
        if (isinstance(pos, np.ndarray) and
                self._ntemps == -1 and
                pos.shape == (nwalkers, nvary)):
            init_walkers = np.copy(pos)[np.newaxis]

        # position is specified with array (with parallel tempering)
        elif (isinstance(pos, np.ndarray) and
              self._ntemps > -1 and
              pos.shape == (_ntemps, nwalkers, nvary)):
            init_walkers = np.copy(pos)

        # position is specified with existing chain
        elif isinstance(pos, np.ndarray):
            self.initialise_with_chain(pos)
            return

        # position is to be created from covariance matrix
        elif pos == 'covar':
            p0 = np.array(self._varying_parameters)
            cov = self.objective.covar()
            init_walkers = emcee.utils.sample_ellipsoid(
                p0,
                cov,
                size=(_ntemps, nwalkers))

        # position is specified by jittering the parameters with gaussian noise
        elif pos == 'jitter':
            var_arr = np.array(self._varying_parameters)
            pos = 1 + np.random.randn(_ntemps,
                                      nwalkers,
                                      nvary) * 1.e-4
            pos *= var_arr
            init_walkers = pos

        # use the prior to initialise position
        elif pos == 'prior':
            arr = np.zeros((_ntemps, nwalkers, nvary))

            for i, param in enumerate(self._varying_parameters):
                # bounds are not a closed interval, just jitter it.
                if (isinstance(param.bounds, Interval) and
                        not param.bounds._closed_bounds):
                    vals = ((1 + np.random.randn(_ntemps, nwalkers) * 1.e-1) *
                            param.value)
                    arr[..., i] = vals
                else:
                    arr[..., i] = param.bounds.rvs(size=(_ntemps, nwalkers))

            init_walkers = arr

        else:
            raise RuntimeError("Didn't use any known method for "
                               "CurveFitter.initialise")

        # if you're not doing parallel tempering then remove the first
        # dimension
        if self._ntemps == -1:
            init_walkers = init_walkers[0]

        # now validate initialisation, ensuring all init pos have finite lnprob
        for i, param in enumerate(self._varying_parameters):
            init_walkers[..., i] = param.valid(init_walkers[..., i])

        self._lastpos = init_walkers

        # finally reset the sampler to reset the chain
        # you have to do this at the end, not at the start because resetting
        # makes self.sampler.chain == None and the PTsampler creation doesn't
        # work
        self.sampler.reset()

    def initialise_with_chain(self, chain):
        """
        Initialise sampler with a pre-existing chain

        Parameters
        ----------
        chain : array
            Array of size `(steps, ntemps, nwalkers, ndim)` or
            `(steps, nwalkers, ndim)`, containing a chain from a previous
            sampling run.
        """
        # we should be left with (nwalkers, ndim) or (ntemp, nwalkers, ndim)
        existing_chain_shape = self.chain.shape[1:]
        chain_shape = chain.shape[1:]

        # if the shapes are the same, then we can initialise
        if existing_chain_shape == chain_shape:
            self.initialise(pos=chain[-1])
        else:
            raise ValueError("You tried to initialise with a chain, but it was"
                             " the wrong shape")

    @property
    def chain(self):
        """
        MCMC chain belonging to CurveFitter.sampler

        Returns
        -------
        chain : array
            The MCMC chain with shape `(steps, nwalkers, ndim)` or
            `(steps, ntemps, nwalkers, ndim)`.

        Notes
        -----
        The chain returned here has swapped axes compared to the
        `PTSampler.chain` and `EnsembleSampler.chain` attributes
        """
        if isinstance(self.sampler, PTSampler):
            return np.transpose(self.sampler.chain, axes=(2, 0, 1, 3))

        return self.sampler.get_chain()

    @property
    def lnprob(self):
        """
        Log-probability for each of the entries in self.chain

        Returns
        -------
        lnprob : array

        """
        if isinstance(self.sampler, PTSampler):
            return np.transpose(self.sampler.logprobability, axes=(2, 0, 1))

        return self.sampler.get_log_prob()

    def acf(self, nburn=0, nthin=1):
        """
        Calculate the autocorrelation function

        Returns
        -------
        acfs : np.ndarray
            The autocorrelation function, acfs.shape=(lags, nvary)
        """
        lchain = self.chain
        if self._ntemps != -1:
            lchain = lchain[:, 0]

        lchain = lchain[nburn::nthin]
        # iterations, walkers, vary
        # (walkers, iterations, vary) -> (vary, walkers, iterations)
        lchain = np.swapaxes(lchain, 0, 2)
        shape = lchain.shape[:-1]

        acfs = np.zeros_like(lchain)

        # iterate over each parameter/walker
        for index in np.ndindex(*shape):
            s = _function_1d(lchain[index])
            acfs[index] = s

        # now average over walkers
        acfs = np.mean(acfs, axis=1)
        return np.transpose(acfs)

    def sample(self, steps, nthin=1, random_state=None, f=None, callback=None,
               verbose=True, pool=0):
        """
        Performs sampling from the objective.

        Parameters
        ----------
        steps : int
            Collect `steps` samples into the chain. The sampler will run a
            total of `steps * nthin` moves.
        nthin : int, optional
            Each chain sample is separated by `nthin` iterations.
        random_state : int or `np.random.RandomState`, optional
            If `random_state` is an int, a new `np.random.RandomState` instance
            is used, seeded with `random_state`.
            If `random_state` is already a `np.random.RandomState` instance,
            then that `np.random.RandomState` instance is used. Specify
            `random_state` for repeatable sampling
        f : file-like or str
            File to incrementally save chain progress to. Each row in the file
            is a flattened array of size `(nwalkers, ndim)` or
            `(ntemps, nwalkers, ndim)`. There are `steps` rows in the
            file.
        callback : callable
            callback function to be called at each iteration step
        verbose : bool, optional
            Gives updates on the sampling progress
        pool : int or map-like object, optional
            If `pool` is an `int` then it specifies the number of threads to
            use for parallelization. If `pool == 0`, then all CPU's are used.
            If pool is an object with a map method that follows the same
            calling sequence as the built-in map function, then this pool is
            used for parallelisation.

        Notes
        -----
        Please see :class:`emcee.EnsembleSampler` for its detailed behaviour.

        >>> # we'll burn the first 500 steps
        >>> fitter.sample(500)
        >>> # after you've run those, then discard them by resetting the
        >>> # sampler.
        >>> fitter.sampler.reset()
        >>> # Now collect 40 steps, each step separated by 50 sampler
        >>> # generations.
        >>> fitter.sample(40, nthin=50)

        One can also burn and thin in `Curvefitter.process_chain`.
        """
        if self._lastpos is None:
            self.initialise()

        self.__pt_iterations = 0
        if isinstance(self.sampler, PTSampler):
            steps *= nthin

        # for saving progress to file
        def _callback_wrapper(pos, lnprob, h=None):
            if callback is not None:
                callback(pos, lnprob)

            if h is not None:
                # if you're parallel tempering, then you only
                # want to save every nthin
                if isinstance(self.sampler, PTSampler):
                    self.__pt_iterations += 1
                    if self.__pt_iterations % nthin:
                        return None

                h.write(' '.join(map(str, pos.ravel())))
                h.write('\n')

        # set the random state of the sampler
        # normally one could give this as an argument to the sample method
        # but PTSampler didn't historically accept that...
        rstate0 = check_random_state(random_state).get_state()
        self.sampler.random_state = rstate0

        # remove chains from each of the parameters because they slow down
        # pickling but only if they are parameter objects.
        flat_params = f_unique(flatten(self.objective.parameters))
        flat_params = [param for param in flat_params if is_parameter(param)]
        # zero out all the old parameter stderrs
        for param in flat_params:
            param.stderr = None
            param.chain = None

        # make sure the checkpoint file exists
        if f is not None:
            with possibly_open_file(f, 'w') as h:
                # write the shape of each step of the chain
                h.write('# ')
                shape = self._lastpos.shape
                h.write(', '.join(map(str, shape)))
                h.write('\n')

        # using context manager means we kill off zombie pool objects
        # but does mean that the pool has to be specified each time.
        with possibly_create_pool(pool) as g, possibly_open_file(f, 'a') as h:
            # if you're not creating more than 1 thread, then don't bother with
            # a pool.
            if pool == 1:
                self.sampler.pool = None
            else:
                self.sampler.pool = g

            kwargs = {'iterations': steps,
                      'thin': nthin}

            # new emcee arguments
            sampler_args = getargspec(self.sampler.sample).args
            if 'progress' in sampler_args and verbose:
                kwargs['progress'] = True
                verbose = False

            if 'thin_by' in sampler_args:
                kwargs['thin_by'] = nthin
                kwargs.pop('thin', 0)

            for result in self.sampler.sample(self._lastpos,
                                              **kwargs):
                self._lastpos, lnprob = result[0:2]
                _callback_wrapper(self._lastpos, lnprob, h=h)

        self.sampler.pool = None

        # finish off the progress bar
        if verbose:
            sys.stdout.write("\n")

        # sets parameter value and stderr
        return process_chain(self.objective, self.chain)

    def fit(self, method='L-BFGS-B', **kws):
        """
        Obtain the maximum log-likelihood estimate (mode) of the objective. For
        a least-squares objective this would correspond to lowest chi2.

        Parameters
        ----------
        method : str
            which method to use for the optimisation. One of:

            - `'least_squares'`: `scipy.optimize.least_squares`.
            - `'L-BFGS-B'`: L-BFGS-B
            - `'differential_evolution'`: differential evolution

            You can also choose many of the minimizers from
            ``scipy.optimize.minimize``.
        kws : dict
            Additional arguments are passed to the underlying minimization
            method.

        Returns
        -------
        result, covar : OptimizeResult, np.ndarray
            `result.x` contains the best fit parameters
            `result.covar` is the covariance matrix for the fit.
            `result.stderr` is the uncertainties on each of the fit parameters.

        Notes
        -----
        If the `objective` supplies a `residuals` method then `least_squares`
        can be used. Otherwise the `nll` method of the `objective` is
        minimised. Use this method just before a sampling run.
        If `self.objective.parameters` is a `Parameters` instance, then each
        of the varying parameters has its value updated by the fit, and each
        `Parameter` has a `stderr` attribute which represents the uncertainty
        on the fit parameter.

        """
        _varying_parameters = self.objective.varying_parameters()
        init_pars = np.array(_varying_parameters)

        _min_kws = {}
        _min_kws.update(kws)
        _bounds = bounds_list(self.objective.varying_parameters())
        _min_kws['bounds'] = _bounds

        # least_squares Trust Region Reflective by default
        if method == 'least_squares':
            b = np.array(_bounds)
            _min_kws['bounds'] = (b[..., 0], b[..., 1])
            res = least_squares(self.objective.residuals,
                                init_pars,
                                **_min_kws)
        # differential_evolution requires lower and upper bounds
        elif method == 'differential_evolution':
            res = differential_evolution(self.objective.nll,
                                         **_min_kws)
        else:
            # otherwise stick it to minimizer. Default being L-BFGS-B
            _min_kws['method'] = method
            _min_kws['bounds'] = _bounds
            res = minimize(self.objective.nll, init_pars, **_min_kws)

        if res.success:
            self.objective.setp(res.x)

            # Covariance matrix estimation
            covar = self.objective.covar()
            errors = np.sqrt(np.diag(covar))
            res['covar'] = covar
            res['stderr'] = errors

            # check if the parameters are all Parameter instances.
            flat_params = list(f_unique(flatten(
                self.objective.parameters)))
            if np.all([is_parameter(param) for param in flat_params]):
                # zero out all the old parameter stderrs
                for param in flat_params:
                    param.stderr = None
                    param.chain = None

                for i, param in enumerate(_varying_parameters):
                    param.stderr = errors[i]

            # need to touch up the output to check we leave
            # parameters as we found them
            self.objective.setp(res.x)

        return res


def load_chain(f):
    """
    Loads a chain from disk. Does not change the state of a CurveFitter
    object.

    Parameters
    ----------
    f : str or file-like
        File containing the chain.

    Returns
    -------
    chain : array
        The loaded chain - `(nsteps, nwalkers, ndim)` or
        `(nsteps, ntemps, nwalkers, ndim)`
    """
    with possibly_open_file(f, 'r') as g:
        # read header
        header = g.readline()
        expr = re.compile('(\d+)')
        matches = expr.findall(header)
        if matches:
            if len(matches) == 3:
                ntemps, nwalkers, ndim = map(int, matches)
                chain_size = ntemps * nwalkers * ndim
            elif len(matches) == 2:
                ntemps = None
                nwalkers, ndim = map(int, matches)
                chain_size = nwalkers * ndim
        else:
            raise ValueError("Couldn't read header line of chain file")

        # make an array that's the appropriate size
        read_arr = array.array("d")

        for i, l in enumerate(g, 1):
            read_arr.extend(np.fromstring(l,
                                          dtype=float,
                                          count=chain_size,
                                          sep=' '))

        chain = np.frombuffer(read_arr, dtype=np.float, count=len(read_arr))

        if ntemps is not None:
            chain = np.reshape(chain, (i, ntemps, nwalkers, ndim))
        else:
            chain = np.reshape(chain, (i, nwalkers, ndim))

        return chain


def process_chain(objective, chain, nburn=0, nthin=1, flatchain=False):
    """
    Process the chain produced by a sampler for a given Objective

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The Objective function that the Posterior was sampled for
    chain : array
        The MCMC chain
    nburn : int, optional
        discard this many steps from the start of the chain
    nthin : int, optional
        only accept every `nthin` samples from the chain
    flatchain : bool, optional
        collapse the walkers down into a single dimension.

    Returns
    -------
    [(param, stderr, chain)] : list
        List of (param, stderr, chain) tuples.
        If `isinstance(objective.parameters, Parameters)` then `param` is a
        `Parameter` instance. `param.value`, `param.stderr` and
        `param.chain` will contain the median, stderr and chain samples,
        respectively. Otherwise `param` will be a float representing the
        median of the chain samples.
        `stderr` is the half width of the [15.87, 84.13] spread (similar to
        standard deviation) and `chain` is an array containing the MCMC
        samples for that parameter.

    Notes
    -----
    The chain should have the shape `(iterations, nwalkers, nvary)` or
    `(iterations, ntemps, nwalkers, nvary)` if parallel tempering was
    employed.
    The burned and thinned chain is created via:
    `chain[nburn::nthin]`.
    Note, if parallel tempering is employed, then only the lowest temperature
    of the parallel tempering chain is processed and returned as it
    corresponds to the (lowest energy) target distribution.
    If `flatten is True` then the burned/thinned chain is reshaped and
    `arr.reshape(-1, nvary)` is returned.
    This function has the effect of setting the parameter stderr's.
    """
    chain = chain[nburn::nthin]
    shape = chain.shape
    nvary = shape[-1]

    # nwalkers = shape[1]
    if len(shape) == 4:
        ntemps = shape[1]
    elif len(shape) == 3:
        ntemps = -1

    if ntemps != -1:
        # PTSampler, we require the target distribution in the first row.
        chain = chain[:, 0]

    _flatchain = chain.reshape((-1, nvary))
    if flatchain:
        chain = _flatchain

    flat_params = list(f_unique(flatten(objective.parameters)))
    varying_parameters = objective.varying_parameters()

    # set the stderr of each of the Parameters
    result_list = []
    if np.all([is_parameter(param) for param in flat_params]):
        # zero out all the old parameter stderrs
        for param in flat_params:
            param.stderr = None
            param.chain = None

        # do the error calcn for the varying parameters and set the chain
        quantiles = np.percentile(_flatchain, [15.87, 50, 84.13], axis=0)
        for i, param in enumerate(varying_parameters):
            std_l, median, std_u = quantiles[:, i]
            param.value = median
            param.stderr = 0.5 * (std_u - std_l)

            # copy in the chain
            param.chain = np.copy(chain[..., i])
            res = MCMCResult(name=param.name, param=param,
                             median=param.value, stderr=param.stderr,
                             chain=param.chain)
            result_list.append(res)

        fitted_values = np.array(varying_parameters)

        # give each constrained param a chain (to be reshaped later)
        constrained_params = [param for param in flat_params
                              if param.constraint is not None]

        for constrain_param in constrained_params:
            constrain_param.chain = np.empty(chain.shape[:-1], float)

        # now iterate through the varying parameters, set the values, thereby
        # setting the constraint value
        for index in np.ndindex(chain.shape[:-1]):
            # iterate over parameter vectors
            pvals = chain[index]
            objective.setp(pvals)

            for constrain_param in constrained_params:
                constrain_param.chain[index] = constrain_param.value

        for constrain_param in constrained_params:
            quantiles = np.percentile(constrain_param.chain,
                                      [15.87, 50, 84.13])

            std_l, median, std_u = quantiles
            constrain_param.value = median
            constrain_param.stderr = 0.5 * (std_u - std_l)

        # now reset fitted parameter values (they would've been changed by
        # constraints calculations
        objective.setp(fitted_values)

    # the parameter set are not Parameter objects, an array was probably
    # being used with BaseObjective.
    else:
        for i in range(nvary):
            c = np.copy(chain[..., i])
            median, stderr = uncertainty_from_chain(c)
            res = MCMCResult(name='', param=median,
                             median=median, stderr=stderr,
                             chain=c)
            result_list.append(res)

    return result_list


def uncertainty_from_chain(chain):
    """
    Calculates the median and uncertainty of MC samples.

    Parameters
    ----------
    chain : array-like

    Returns
    -------
    median, stderr : float, float
        `median` of the chain samples. `stderr` is half the width of the
        [15.87, 84.13] spread.
    """
    flatchain = chain.flatten()
    std_l, median, std_u = np.percentile(flatchain, [15.87, 50, 84.13])
    return median, 0.5 * (std_u - std_l)


def bounds_list(parameters):
    """
    Return (interval) bounds for all varying parameters
    """
    bounds = []
    for param in parameters:
        if (hasattr(param, 'bounds') and
                isinstance(param.bounds, Interval)):
            bnd = param.bounds
            bounds.append((bnd.lb, bnd.ub))
        # TODO could also do any truncated PDF

    return bounds


# Following code is for autocorrelation analysis of chains and is taken from
# emcee.autocorr
def _next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def _function_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series

    Args:
        x: The series as a 1-D numpy array.

    Returns:
        array: The autocorrelation function of the time series.

    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = _next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= acf[0]
    return acf
