from collections import namedtuple
import sys
import re
import warnings
import array

import numpy as np

from scipy._lib._util import check_random_state
from scipy.optimize import minimize, differential_evolution, least_squares
import scipy.optimize as sciopt

from refnx.analysis import Objective, Interval, PDF, is_parameter
from refnx._lib import (
    unique as f_unique,
    MapWrapper,
    possibly_open_file,
    flatten,
)
from refnx._lib.util import getargspec
from refnx._lib._qmc import LatinHypercube
from refnx._lib import emcee
from refnx._lib.emcee.state import State
from refnx._lib.emcee.pbar import get_progress_bar


MCMCResult = namedtuple(
    "MCMCResult", ["name", "param", "stderr", "chain", "median"]
)


class PTSampler:
    def __init__(self, ntemps, nwalkers, ndim, logl, logp, **kwargs):
        """
        Shim class for a ptemcee.PTSampler.

        Parameters
        ----------
        ntemps: int, np.array
            Specifies the number of parallel tempering temperatures.
            If an array specifies a ladder of Beta values.
        nwalkers: int
            Number of walkers
        ndim: int
            Dimensionality of the problem space
        logl: callable
            log-likelihood function
        logp: callable
            log-prior function
        kwargs:
            Other keyword arguments supplied to construct the ptemcee.Sampler.
        """
        from refnx._lib.ptemcee import Sampler as _PTSampler

        self.ntemps = ntemps
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.logl = logl
        self.logp = logp
        self.kwargs = kwargs

        sig = {
            "betas": ntemps,
            "nwalkers": nwalkers,
            "ndim": ndim,
            "logl": logl,
            "logp": logp,
        }
        sig.update(kwargs)
        self.sampler = _PTSampler(**sig)

        # chain stepper
        self._ptchain = None
        self._state = None

    def sample(
        self,
        initial_state,
        iterations=1,
        thin_by=1,
        progress=False,
        mapper=None,
        **kwds,
    ):
        """
        Runs the PTSampler for a given number of iterations.

        Parameters
        ----------
        initial_state: emcee.state.State
            Holds the coordinates of the initial state
        iterations: int
            Number of steps to save into the chain
        thin_by: int
            The saved steps are separated by this many discarded steps.
        progress: bool
            Display a progress bar.
        mapper: map-like callable
            For parallelisation
        kwds: dict
            Unknown keywords

        Yields
        -------
        state: emcee.state.State
            The coordinates of the current state
        """
        if isinstance(initial_state, State):
            init_x = initial_state.coords
            rstate0 = initial_state.random_state
        else:
            init_x = initial_state
            rstate0 = np.random.RandomState().get_state()

        if self._ptchain is None:
            self._ptchain = self.sampler.chain(init_x)
        else:
            self._ptchain.ensemble.x = init_x

        # set random state of stateful chain
        self.random_state = rstate0

        self._ptchain.thin_by = thin_by

        if mapper is not None:
            self._ptchain.ensemble._mapper = mapper

        try:
            with get_progress_bar(progress, iterations * thin_by) as pbar:
                for e in self._ptchain.iterate(iterations):
                    self._state = State(e.x, log_prob=e.logl + e.logP)
                    yield self._state
                    pbar.update(thin_by)
        finally:
            self._ptchain.ensemble._mapper = map

    def thermodynamic_integration_log_evidence(self, fburnin=0.1):
        if self._ptchain is not None:
            return self._ptchain.log_evidence_estimate(fburnin)
        return None, None

    def reset(self):
        if self._state is not None:
            self._ptchain = self.sampler.chain(self._state.coords)

    def get_chain(self):
        if self._ptchain is not None:
            return self._ptchain.x
        return None

    @property
    def chain(self):
        if self._ptchain is not None:
            return self._ptchain.x
        return None

    def get_log_prob(self):
        if self._ptchain is not None:
            return self._ptchain.logP
        return None

    @property
    def random_state(self):
        if self._ptchain is not None:
            self._ptchain.ensemble._random.get_state()

    @random_state.setter
    def random_state(self, rstate0):
        if self._ptchain is not None:
            self._ptchain.ensemble._random.set_state(rstate0)


class CurveFitter:
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
        self._varying_parameters = []
        self.__var_id = []

        self.mcmc_kws = {}
        if mcmc_kws is not None:
            self.mcmc_kws.update(mcmc_kws)

        if "pool" in self.mcmc_kws:
            self.mcmc_kws.pop("pool")
        if "threads" in self.mcmc_kws:
            self.mcmc_kws.pop("threads")

        self._nwalkers = nwalkers
        self._ntemps = ntemps
        self.make_sampler()
        self._state = None

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__var_id = [
            id(obj) for obj in self.objective.varying_parameters()
        ]

    @property
    def nvary(self):
        return len(self._varying_parameters)

    def __repr__(self):
        # attempt to get a minimum repr for a CurveFitter. However,
        # it has so much state when the sampling has been done, that
        # will be ignored.
        d = {
            "objective": self.objective,
            "_nwalkers": self._nwalkers,
            "_ntemps": self._ntemps,
            "mcmc_kws": self.mcmc_kws,
        }
        return (
            "CurveFitter({objective!r},"
            " nwalkers={_nwalkers},"
            " ntemps={_ntemps},"
            " {mcmc_kws!r})".format(**d)
        )

    def make_sampler(self):
        """
        Make the samplers for the Objective.

        Use this method if the number of varying parameters changes.
        """
        self._varying_parameters = self.objective.varying_parameters()
        self.__var_id = [id(obj) for obj in self._varying_parameters]

        if not self.nvary:
            raise ValueError("No parameters are being fitted")

        if self._ntemps == -1:
            self.sampler = emcee.EnsembleSampler(
                self._nwalkers,
                self.nvary,
                self.objective.logpost,
                **self.mcmc_kws,
            )
        # Parallel Tempering was requested.
        else:
            sig = {
                "ntemps": self._ntemps,
                "nwalkers": self._nwalkers,
                "ndim": self.nvary,
                "logl": self.objective.logl,
                "logp": self.objective.logp,
            }
            sig.update(self.mcmc_kws)
            self.sampler = PTSampler(**sig)

        self._state = None

    def _check_vars_unchanged(self):
        """
        Keep track of whether the varying parameters have changed after
        construction of CurveFitter object.

        """
        var_ids = [id(obj) for obj in self.objective.varying_parameters()]
        if not (np.array_equal(var_ids, self.__var_id)):
            raise RuntimeError(
                "The Objective.varying_parameters() have"
                " changed since the CurveFitter was created."
                " To keep on using the CurveFitter call"
                " the CurveFitter.make_samplers() method."
            )

    def initialise(self, pos="covar", random_state=None):
        """
        Initialise the emcee walkers.

        Parameters
        ----------
        pos : str or np.ndarray
            Method for initialising the emcee walkers. One of:

            - 'covar', use the estimated covariance of the system.
            - 'jitter', add a small amount of gaussian noise to each parameter
            - 'prior', sample random locations from the prior using Latin
                Hyper Cube.
            - pos, an array that specifies a snapshot of the walkers. Has shape
                `(nwalkers, ndim)`, or `(ntemps, nwalkers, ndim)` if parallel
                 tempering is employed. You can also provide a previously
                 created chain.
        random_state : {int, `np.random.RandomState`, `np.random.Generator`}
            If `random_state` is not specified the `~np.random.RandomState`
            singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is
            used, seeded with random_state.
            If `random_state` is already a ``RandomState`` or a ``Generator``
            instance, then that object is used.
            Specify `random_state` for repeatable initialisations.
        """
        nwalkers = self._nwalkers
        nvary = self.nvary

        # acquire a random number generator
        rng = check_random_state(random_state)

        # account for parallel tempering
        _ntemps = self._ntemps

        # If you're not doing parallel tempering, temporarily set the number of
        # temperatures to be created to 1, thereby producing initial positions
        # of (1, nwalkers, nvary), this first dimension should be removed at
        # the end of the method
        if self._ntemps == -1:
            _ntemps = 1

        # position is specified with array (no parallel tempering)
        if (
            isinstance(pos, np.ndarray)
            and self._ntemps == -1
            and pos.shape == (nwalkers, nvary)
        ):
            init_walkers = np.copy(pos)[np.newaxis]

        # position is specified with array (with parallel tempering)
        elif (
            isinstance(pos, np.ndarray)
            and self._ntemps > -1
            and pos.shape == (_ntemps, nwalkers, nvary)
        ):
            init_walkers = np.copy(pos)

        # position is specified with existing chain
        elif isinstance(pos, np.ndarray):
            self.initialise_with_chain(pos)
            return

        # position is to be created from covariance matrix
        elif pos == "covar":
            p0 = np.array(self._varying_parameters)
            cov = self.objective.covar()
            init_walkers = rng.multivariate_normal(
                np.atleast_1d(p0), np.atleast_2d(cov), size=(_ntemps, nwalkers)
            )

        # position is specified by jittering the parameters with gaussian noise
        elif pos == "jitter":
            var_arr = np.array(self._varying_parameters)
            pos = 1 + rng.standard_normal((_ntemps, nwalkers, nvary)) * 1.0e-4
            pos *= var_arr
            init_walkers = pos

        # use the prior to initialise position
        elif pos == "prior":
            arr = np.zeros((_ntemps, nwalkers, nvary))
            LHC = LatinHypercube(nvary, seed=random_state)
            samples = LHC.random(n=_ntemps * nwalkers).reshape(
                _ntemps, nwalkers, nvary
            )

            for i, param in enumerate(self._varying_parameters):
                # bounds are not a closed interval, just jitter it.
                if (
                    isinstance(param.bounds, Interval)
                    and not param.bounds._closed_bounds
                ):
                    vals = (
                        1 + rng.standard_normal((_ntemps, nwalkers)) * 1.0e-1
                    )
                    vals *= param.value
                    arr[..., i] = vals
                else:
                    sample_arr = samples[..., i]
                    transformed = param.bounds.invcdf(sample_arr)
                    arr[..., i] = transformed

            init_walkers = arr

        else:
            raise RuntimeError(
                "Didn't use any known method for " "CurveFitter.initialise"
            )

        # if you're not doing parallel tempering then remove the first
        # dimension
        if self._ntemps == -1:
            init_walkers = init_walkers[0]

        # now validate initialisation, ensuring all init pos have finite
        # logpost
        for i, param in enumerate(self._varying_parameters):
            init_walkers[..., i] = param.valid(init_walkers[..., i])

        rstate0 = None
        if isinstance(rng, np.random.RandomState):
            rstate0 = rng.get_state()

        self._state = State(init_walkers, random_state=rstate0)

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

        if self._ntemps == -1:
            required_shape = (self._nwalkers, self.nvary)
        else:
            required_shape = (self._ntemps, self._nwalkers, self.nvary)

        chain_shape = chain.shape[1:]

        # if the shapes are the same, then we can initialise
        if required_shape == chain_shape:
            self.initialise(pos=chain[-1])
        else:
            raise ValueError(
                "You tried to initialise with a chain, but it was"
                " the wrong shape"
            )

    @property
    def chain(self):
        """
        MCMC chain belonging to CurveFitter.sampler

        Returns
        -------
        chain : array
            The MCMC chain with shape `(steps, nwalkers, ndim)` or
            `(steps, ntemps, nwalkers, ndim)`.
        """
        return self.sampler.get_chain()

    @property
    def logpost(self):
        """
        Log-probability for each of the entries in `self.chain`
        """
        return self.sampler.get_log_prob()

    @property
    def index_max_prob(self):
        """
        The index of the highest log-probability for the samples
        """
        log_probs = self.sampler.get_log_prob()
        if isinstance(self.sampler, PTSampler):
            log_probs = log_probs[:, 0]

        loc = np.argmax(log_probs)
        idx = np.unravel_index(loc, log_probs.shape)

        if isinstance(self.sampler, PTSampler):
            idx = list(idx)
            idx.insert(1, 0)
            return tuple(idx)

        return idx

    def reset(self):
        """
        Reset the sampled chain.

        Typically used on a sampler after a burn-in period.
        """
        self.sampler.reset()

    def acf(self, nburn=0, nthin=1):
        """
        Calculate the autocorrelation function

        Returns
        -------
        acfs : np.ndarray
            The autocorrelation function, acfs.shape=(lags, nvary)
        """
        return autocorrelation_chain(self.chain, nburn=nburn, nthin=nthin)

    def sample(
        self,
        steps,
        nthin=1,
        random_state=None,
        f=None,
        callback=None,
        verbose=True,
        pool=-1,
    ):
        """
        Performs sampling from the objective.

        Parameters
        ----------
        steps : int
            Collect `steps` samples into the chain. The sampler will run a
            total of `steps * nthin` moves.
        nthin : int, optional
            Each chain sample is separated by `nthin` iterations.
        random_state : {int, `np.random.RandomState`, `np.random.Generator`}
            If `random_state` is not specified the `~np.random.RandomState`
            singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is
            used, seeded with random_state.
            If `random_state` is already a ``RandomState`` or a ``Generator``
            instance, then that object is used.
            Specify `random_state` for repeatable minimizations.
        f : file-like or str
            File to incrementally save chain progress to. Each row in the file
            is a flattened array of size `(nwalkers, ndim)` or
            `(ntemps, nwalkers, ndim)`. There are `steps` rows in the
            file.
        callback : callable
            callback function to be called at each iteration step. Has the
            signature `callback(coords, logprob)`.
        verbose : bool, optional
            Gives updates on the sampling progress
        pool : int or map-like object, optional
            If `pool` is an `int` then it specifies the number of threads to
            use for parallelization. If `pool == -1`, then all CPU's are used.
            If pool is a map-like callable that follows the same calling
            sequence as the built-in map function, then this pool is used for
            parallelisation.

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
        self._check_vars_unchanged()

        # setup a random number generator
        rng = check_random_state(random_state)

        if self._state is None:
            self.initialise(random_state=rng)

        # for saving progress to file
        def _callback_wrapper(state, h=None):
            if callback is not None:
                callback(state.coords, state.log_prob)

            if h is not None:
                h.write(" ".join(map(str, state.coords.ravel())))
                h.write("\n")

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
            with possibly_open_file(f, "w") as h:
                # write the shape of each step of the chain
                h.write("# ")
                shape = self._state.coords.shape
                h.write(", ".join(map(str, shape)))
                h.write("\n")

        # set the random state of the sampler
        # normally one could give this as an argument to the sample method
        # but PTSampler didn't historically accept that...
        if isinstance(rng, np.random.RandomState):
            rstate0 = rng.get_state()
            self._state.random_state = rstate0
            self.sampler.random_state = rstate0

        # using context manager means we kill off zombie pool objects
        # but does mean that the pool has to be specified each time.
        with MapWrapper(pool) as g, possibly_open_file(f, "a") as h:
            # these kwargs are provided to the sampler.sample method
            kwargs = {"iterations": steps, "thin": nthin}

            # if you're not creating more than 1 thread, then don't bother with
            # a pool.
            if isinstance(self.sampler, emcee.EnsembleSampler):
                if pool == 1:
                    self.sampler.pool = None
                else:
                    self.sampler.pool = g
            else:
                kwargs["mapper"] = g

            # new emcee arguments
            sampler_args = getargspec(self.sampler.sample).args
            if "progress" in sampler_args and verbose:
                kwargs["progress"] = True
                verbose = False

            if "thin_by" in sampler_args:
                kwargs["thin_by"] = nthin
                kwargs.pop("thin", 0)

            # perform the sampling
            for state in self.sampler.sample(self._state, **kwargs):
                self._state = state
                _callback_wrapper(state, h=h)

        if isinstance(self.sampler, emcee.EnsembleSampler):
            self.sampler.pool = None

        # sets parameter value and stderr
        return process_chain(self.objective, self.chain)

    def fit(self, method="L-BFGS-B", target="nll", verbose=True, **kws):
        """
        Obtain the maximum log-likelihood, or log-posterior, estimate (mode)
        of the objective. Maximising the log-likelihood is equivalent to
        minimising chi2 in a least squares fit.

        Parameters
        ----------
        method : str
            which method to use for the optimisation. One of:

            - `'least_squares'`: :func:`scipy.optimize.least_squares`.
            - `'L-BFGS-B'`: L-BFGS-B.
            - `'differential_evolution'`:
              :func:`scipy.optimize.differential_evolution`
            - `'dual_annealing'`:
              :func:`scipy.optimize.dual_annealing` (SciPy >= 1.2.0)
            - `'shgo'`: :func:`scipy.optimize.shgo` (SciPy >= 1.2.0)

            You can also choose many of the minimizers from
            :func:`scipy.optimize.minimize`.

        target : {'nll', 'nlpost'}, optional
            Minimize the negative log-likelihood (`'nll'`) or the negative
            log-posterior (`'nlpost'`). This is equivalent to maximising the
            likelihood or posterior probabilities respectively.
            Maximising the likelihood is equivalent to minimising chi^2 in a
            least-squares fit.
            This option only applies to the `differential_evolution`, `shgo`,
            `dual_annealing` or `L-BFGS-B` methods.
            These optimisers require lower and upper (box) bounds for each
            parameter. If the `Bounds` on a parameter are not an `Interval`,
            but a `PDF` specifying a statistical distribution, then the lower
            and upper bounds are approximated as
            ``PDF.rv.ppf([0.005, 0.995])``, covering 99 % of the statistical
            distribution.
        verbose : bool, optional
            Gives fitting progress. To see a progress bar tqdm has to be
            installed.
        kws : dict
            Additional arguments are passed to the underlying minimization
            method.

        Returns
        -------
        result, covar : :class:`scipy.optimize.OptimizeResult`, np.ndarray
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

        The use of `dual annealing` and `shgo` requires that `scipy >= 1.2.0`
        be installed.

        """
        _varying_parameters = self.objective.varying_parameters()
        init_pars = np.array(_varying_parameters)

        _min_kws = {}
        _min_kws.update(kws)
        _bounds = bounds_list(self.objective.varying_parameters())
        _min_kws["bounds"] = _bounds

        # setup callback default
        _min_kws.setdefault("callback", None)

        cost = self.objective.nll
        if target == "nlpost":
            cost = self.objective.nlpost

        # a decorator for the progress bar updater
        def _callback_wrapper(callback_func, pbar):
            def callback(*args, **kwds):
                pbar.update(1)
                if callback_func is None:
                    return None
                else:
                    return callback_func(*args, **kwds)

            return callback

        # least_squares Trust Region Reflective by default
        if method == "least_squares":
            b = np.array(_bounds)
            _min_kws["bounds"] = (b[..., 0], b[..., 1])

            # least_squares doesn't have a callback
            _min_kws.pop("callback", None)

            res = least_squares(
                self.objective.residuals, init_pars, **_min_kws
            )
        # differential_evolution, dual_annealing, shgo require lower and upper
        # bounds
        elif method in ["differential_evolution", "dual_annealing", "shgo"]:
            mini = getattr(sciopt, method)

            if method == "shgo":
                if "n" not in _min_kws:
                    _min_kws["n"] = 100
                if "iters" not in kws:
                    _min_kws["iters"] = 5

            with get_progress_bar(verbose, None) as pbar:
                _min_kws["callback"] = _callback_wrapper(
                    _min_kws["callback"], pbar
                )

                res = mini(cost, **_min_kws)
        else:
            # otherwise stick it to minimizer. Default being L-BFGS-B
            _min_kws["method"] = method
            _min_kws["bounds"] = _bounds

            with get_progress_bar(verbose, None) as pbar:
                _min_kws["callback"] = _callback_wrapper(
                    _min_kws["callback"], pbar
                )

                res = minimize(cost, init_pars, **_min_kws)

        # OptimizeResult.success may not be present (dual annealing)
        if hasattr(res, "success") and res.success:
            self.objective.setp(res.x)

            # Covariance matrix estimation
            covar = self.objective.covar()
            errors = np.sqrt(np.diag(covar))
            res["covar"] = covar
            res["stderr"] = errors

            # check if the parameters are all Parameter instances.
            flat_params = list(f_unique(flatten(self.objective.parameters)))
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
    with possibly_open_file(f, "r") as g:
        # read header
        header = g.readline()
        expr = re.compile(r"(\d+)")
        matches = expr.findall(header)
        if matches:
            if len(matches) == 3:
                ntemps, nwalkers, ndim = map(int, matches)
            elif len(matches) == 2:
                ntemps = None
                nwalkers, ndim = map(int, matches)
        else:
            raise ValueError("Couldn't read header line of chain file")

    chain = np.loadtxt(f)

    if ntemps is not None:
        chain = np.reshape(chain, (-1, ntemps, nwalkers, ndim))
    else:
        chain = np.reshape(chain, (-1, nwalkers, ndim))

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
            res = MCMCResult(
                name=param.name,
                param=param,
                median=param.value,
                stderr=param.stderr,
                chain=param.chain,
            )
            result_list.append(res)

        fitted_values = np.array(varying_parameters)

        # give each constrained param a chain (to be reshaped later)
        constrained_params = [
            param for param in flat_params if param.constraint is not None
        ]

        for constrain_param in constrained_params:
            constrain_param.chain = np.empty(chain.shape[:-1], float)

        # now iterate through the varying parameters, set the values, thereby
        # setting the constraint value
        if len(constrained_params):
            for index in np.ndindex(chain.shape[:-1]):
                # iterate over parameter vectors
                pvals = chain[index]
                objective.setp(pvals)

                for constrain_param in constrained_params:
                    constrain_param.chain[index] = constrain_param.value

            for constrain_param in constrained_params:
                quantiles = np.percentile(
                    constrain_param.chain, [15.87, 50, 84.13]
                )

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
            res = MCMCResult(
                name="", param=median, median=median, stderr=stderr, chain=c
            )
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


def autocorrelation_chain(chain, nburn=0, nthin=1):
    """
    Calculate the autocorrelation function

    Parameters
    ----------
    chain : np.ndarray
        The MCMC chain - `(nsteps, nwalkers, ndim)` or
        `(nsteps, ntemps, nwalkers, ndim)`

    Returns
    -------
    acfs : np.ndarray
        The autocorrelation function, acfs.shape=(lags, nvary)
    """
    lchain = chain
    # parallel tempered chain
    if len(chain.shape) == 4:
        lchain = lchain[:, 0]

    lchain = lchain[nburn::nthin]
    # (iterations, walkers, vary) -> (vary, walkers, iterations)
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


def bounds_list(parameters):
    """
    Approximates interval bounds for a parameter set.

    Parameters
    ----------
    parameters : sequence
        A sequence containing individual parameters

    Returns
    -------
    bounds: tuple
        ``(min, max)`` pairs that define the finite lower and upper bounds
        every element in ``parameters``.

    If the `Bounds` applied by a parameter are a `PDF` instance then the upper
    and lower bound are approximated by ``PDF.rv.ppf([0.005, 0.995])``, which
    covers 99% of the statistical distribution.
    """
    bounds = []
    for param in parameters:
        if hasattr(param, "bounds") and isinstance(param.bounds, Interval):
            bnd = param.bounds
            bounds.append((bnd.lb, bnd.ub))
        elif (
            hasattr(param, "bounds")
            and isinstance(param.bounds, PDF)
            and hasattr(param.bounds.rv, "ppf")
        ):
            bounds.append(param.bounds.rv.ppf([0.005, 0.995]))
        else:
            # We can't handle this bound
            bounds.append((-np.inf, np.inf))

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
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= acf[0]
    return acf
