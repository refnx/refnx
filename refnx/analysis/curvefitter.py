from __future__ import division

from functools import partial
from collections import namedtuple
import sys
import time

import numpy as np
import emcee as emcee
from scipy._lib._util import check_random_state
from scipy.optimize import minimize, differential_evolution, least_squares

from refnx.analysis import Objective, Interval, PDF, is_parameter
from refnx._lib import flatten
from refnx._lib import unique as f_unique, possibly_create_pool

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


class CurveFitter(object):
    """
    Analyse a curvefitting system (with MCMC sampling)
    """
    def __init__(self, objective, nwalkers=200, **mcmc_kws):
        """
        Parameters
        ----------
        objective : Objective
            The :class:`playtime.objective.Objective` to be analysed.
        nwalkers : int
            How many walkers you would like the sampler to have. Must be an
            even number. The more walkers the better.
        mcmc_kws : dict
            Keywords used to create the :class:`emcee.EnsembleSampler` object.

        Notes
        -----
        See the documentation at http://dan.iel.fm/emcee/current/api/ for
        further details on what keywords are permitted. The `pool` and
        `threads` keywords are ignored here. Specification of parallel
        threading is done with the `pool` argument in the `sample` method.
        """
        self.objective = objective
        self._varying_parameters = objective.varying_parameters()
        self.nvary = len(self._varying_parameters)
        if not self.nvary:
            raise ValueError("No parameters are being fitted")

        self.mcmc_kws = {}
        if mcmc_kws is not None:
            self.mcmc_kws.update(mcmc_kws)

        self.mcmc_kws['args'] = (objective,)

        if 'pool' in self.mcmc_kws:
            self.mcmc_kws.pop('pool')
        if 'threads' in self.mcmc_kws:
            self.mcmc_kws.pop('threads')

        self._nwalkers = nwalkers
        self.sampler = emcee.EnsembleSampler(nwalkers,
                                             self.nvary,
                                             _objective_lnprob,
                                             **self.mcmc_kws)
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
                `(nwalkers, ndim)`.
        """
        self.sampler.reset()
        if pos == 'covar':
            p0 = np.array(self._varying_parameters)
            nwalkers = self._nwalkers
            cov = self.objective.covar()
            self._lastpos = emcee.utils.sample_ellipsoid(p0,
                                                         cov,
                                                         size=nwalkers)
        elif (isinstance(pos, np.ndarray) and
              pos.shape == (self._nwalkers,
                            self.nvary)):
            self._lastpos = pos

        elif pos == 'jitter':
            var_arr = np.array(self._varying_parameters)
            pos = 1 + np.random.randn(self._nwalkers,
                                      self.nvary) * 1.e-4
            pos *= var_arr
            self._lastpos = pos

        elif pos == 'prior':
            arr = np.zeros((self._nwalkers, self.nvary))

            for i, param in enumerate(self._varying_parameters):
                # bounds are not a closed interval, just jitter it.
                if (isinstance(param.bounds, Interval) and
                        not param.bounds._closed_bounds):
                    vals = ((1 + np.random.randn(self._nwalkers) * 1.e-1) *
                            param.value)
                    arr[:, i] = vals
                else:
                    arr[:, i] = param.bounds.rvs(size=self._nwalkers)
            self._lastpos = arr

        else:
            raise RuntimeError("Didn't initialise CurveFitter with any known"
                               " method.")

        # now validate initialisation, ensuring all init pos have finite lnprob
        for i, param in enumerate(self._varying_parameters):
            self._lastpos[..., i] = param.valid(self._lastpos[..., i])

    def sample(self, steps, nburn=0, nthin=1, random_state=None, f=None,
               callback=None, verbose=True, pool=0):
        """
        Performs sampling from the objective.

        Parameters
        ----------
        steps : int
            Iterate the sampler by a number of steps
        nburn : int, optional
            discard this many steps from the chain
        nthin : int, optional
            only accept every `nthin` samples from the chain
        random_state : int or `np.random.RandomState`, optional
            If `random_state` is an int, a new `np.random.RandomState` instance
            is used, seeded with `random_state`.
            If `random_state` is already a `np.random.RandomState` instance,
            then that `np.random.RandomState` instance is used. Specify
            `random_state` for repeatable sampling
        f : file-like or str
            File to incrementally save chain progress to
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
        For example, the chain is contained in `CurveFitter.sampler.chain` and
        has shape `(nwalkers, iterations, ndim)`. `nsteps` should be greater
        than `nburn`.
        """
        if self._lastpos is None:
            self.initialise()

        start_time = time.time()

        # for saving progress to file, and printing progress to stdout.
        def _callback_wrapper(pos, lnprob, steps_completed):
            if verbose:
                steps_completed += 1
                width = 50
                step_rate = (time.time() - start_time) / steps_completed
                time_remaining = divmod(step_rate * (steps - steps_completed),
                                        60)
                mins, secs = int(time_remaining[0]), int(time_remaining[1])
                n = int((width + 1) * float(steps_completed) / steps)
                template = ("\rSampling progress: [{0}{1}] "
                            "time remaining = {2} m:{3} s")
                sys.stdout.write(template.format('#' * n,
                                                 ' ' * (width - n),
                                                 mins,
                                                 secs))
            if callback is not None:
                callback(pos, lnprob)
            if f is not None:
                np.save(f, self.sampler.chain)

        rstate0 = check_random_state(random_state).get_state()

        # remove chains from each of the parameters because they slow down
        # pickling but only if they are parameter objects.
        flat_params = f_unique(flatten(self.objective.parameters))
        flat_params = [param for param in flat_params if is_parameter(param)]
        # zero out all the old parameter stderrs
        for param in flat_params:
            param.stderr = None
            param.chain = None

        # using context manager means we kill off zombie pool objects
        # but does mean that the pool has to be specified each time.
        with possibly_create_pool(pool) as g:
            # if you're not creating more than 1 thread, then don't bother with
            # a pool.
            if pool == 1:
                self.sampler.pool = None
            else:
                self.sampler.pool = g

            for i, result in enumerate(self.sampler.sample(self._lastpos,
                                                           iterations=steps,
                                                           rstate0=rstate0)):
                pos, lnprob = result[0:2]
                _callback_wrapper(pos, lnprob, i)

        self.sampler.pool = None
        self._lastpos = result[0]

        # finish off the progress bar
        if verbose:
            sys.stdout.write("\n")

        # sets parameter value and stderr
        return self.process_chain(nburn=nburn, nthin=nthin)

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

            # Covariance from numdifftools seems to be pretty reliable
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

            # need to touch up the output as numdifftools doesn't leave
            # parameters as it found them
            self.objective.setp(res.x)

        return res

    def process_chain(self, nburn=0, nthin=1, flatchain=False):
        """
        Process the chain produced by the sampler.

        Parameters
        ----------
        nburn : int, optional
            discard this many steps from the chain
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
        One can call `process_chain` many times, the chain associated with this
        object is unaltered. The chain is stored in the
        `CurveFitter.sampler.chain` attribute and has shape
        `(nwalkers, iterations, nvary)`. The burned and thinned chain is
        created via: `chain[:, nburn::nthin]`. If `flatten is True` then
        `chain[:, nburn::nthin].reshape(-1, nvary) is returned. It also has the
        effect of setting the parameter stderr's.
        """
        chain = self.sampler.chain[:, nburn::nthin, :]
        _flatchain = chain.reshape((-1, self.nvary))
        if flatchain:
            chain = _flatchain

        flat_params = list(f_unique(flatten(self.objective.parameters)))

        # set the stderr of each of the Parameters
        l = []
        if np.all([is_parameter(param) for param in flat_params]):
            # zero out all the old parameter stderrs
            for param in flat_params:
                param.stderr = None
                param.chain = None

            # do the error calcn for the varying parameters and set the chain
            quantiles = np.percentile(_flatchain, [15.87, 50, 84.13], axis=0)
            for i, param in enumerate(self._varying_parameters):
                std_l, median, std_u = quantiles[:, i]
                param.value = median
                param.stderr = 0.5 * (std_u - std_l)

                # copy in the chain
                param.chain = np.copy(chain[..., i])
                res = MCMCResult(name=param.name, param=param,
                                 median=param.value, stderr=param.stderr,
                                 chain=param.chain)
                l.append(res)

            fitted_values = np.array(self._varying_parameters)

            # give each constrained param a chain (to be reshaped later)
            # but only if it depends on varying parameters
            # TODO add a test for this...
            constrained_params = [param for param in flat_params
                                  if param.constraint is not None]

            # figure out all the "master" parameters for the constrained
            # parameters
            relevant_depends = []
            for constrain_param in constrained_params:
                depends = set(flatten(constrain_param.dependencies))
                # we only need the dependencies that are varying parameters
                rdepends = depends.intersection(set(self._varying_parameters))
                relevant_depends.append(rdepends)

            # don't need duplicates
            relevant_depends = set(relevant_depends)

            for constrain_param in constrained_params:
                depends = set(flatten(constrain_param.dependencies))
                # to be given a chain the constrained parameter has to depend
                # on a varying parameter
                if depends.intersection(relevant_depends):
                    constrain_param.chain = np.zeros_like(
                        relevant_depends[0].chain)

                    for index, _ in np.ndenumerate(constrain_param.chain):
                        for rdepend in relevant_depends:
                            rdepend.value = rdepend.chain[index]

                        constrain_param.chain[index] = constrain_param.value

                    quantiles = np.percentile(constrain_param.chain,
                                              [15.87, 50, 84.13])

                    std_l, median, std_u = quantiles
                    constrain_param.value = median
                    constrain_param.stderr = 0.5 * (std_u - std_l)

            # now reset fitted parameter values (they would've been changed by
            # constraints calculations
            self.objective.setp(fitted_values)
        else:
            for i in range(self.nvary):
                c = np.copy(chain[..., i])
                median, stderr = uncertainty_from_chain(c)
                res = MCMCResult(name='', param=median,
                                 median=median, stderr=stderr,
                                 chain=c)
                l.append(res)

        return l


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
