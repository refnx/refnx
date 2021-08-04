import os.path
import pickle

import numpy as np
import scipy.optimize as sciopt
from scipy.stats import norm

import pytest
from numpy.testing import (
    assert_,
    assert_almost_equal,
    assert_equal,
    assert_allclose,
)

from refnx.analysis import (
    CurveFitter,
    Parameter,
    Parameters,
    Model,
    Objective,
    process_chain,
    load_chain,
    Bounds,
    PDF,
    autocorrelation_chain,
    integrated_time,
)
from refnx.analysis.curvefitter import bounds_list
from refnx.dataset import Data1D
from refnx._lib import emcee, flatten

from NISTModels import NIST_runner, NIST_Models


def line(x, params, *args, **kwds):
    p_arr = np.array(params)
    return p_arr[0] + x * p_arr[1]


class TestCurveFitter:
    def setup_method(self):
        # Reproducible results!
        np.random.seed(123)

        self.m_true = -0.9594
        self.b_true = 4.294
        self.f_true = 0.534
        self.m_ls = -1.1040757010910947
        self.b_ls = 5.4405552502319505

        # Generate some synthetic data from the model.
        N = 50
        x = np.sort(10 * np.random.rand(N))
        y_err = 0.1 + 0.5 * np.random.rand(N)
        y = self.m_true * x + self.b_true
        y += np.abs(self.f_true * y) * np.random.randn(N)
        y += y_err * np.random.randn(N)

        self.data = Data1D(data=(x, y, y_err))

        self.p = Parameter(self.b_ls, "b", vary=True, bounds=(-100, 100))
        self.p |= Parameter(self.m_ls, "m", vary=True, bounds=(-100, 100))

        self.model = Model(self.p, fitfunc=line)
        self.objective = Objective(self.model, self.data)
        assert_(len(self.objective.varying_parameters()) == 2)

        mod = np.array(
            [
                4.78166609,
                4.42364699,
                4.16404064,
                3.50343504,
                3.4257084,
                2.93594347,
                2.92035638,
                2.67533842,
                2.28136038,
                2.19772983,
                1.99295496,
                1.93748334,
                1.87484436,
                1.65161016,
                1.44613461,
                1.11128101,
                1.04584535,
                0.86055984,
                0.76913963,
                0.73906649,
                0.73331407,
                0.68350418,
                0.65216599,
                0.59838566,
                0.13070299,
                0.10749131,
                -0.01010195,
                -0.10010155,
                -0.29495372,
                -0.42817431,
                -0.43122391,
                -0.64637715,
                -1.30560686,
                -1.32626428,
                -1.44835768,
                -1.52589881,
                -1.56371158,
                -2.12048349,
                -2.24899179,
                -2.50292682,
                -2.53576659,
                -2.55797996,
                -2.60870542,
                -2.7074727,
                -3.93781479,
                -4.12415366,
                -4.42313742,
                -4.98368609,
                -5.38782395,
                -5.44077086,
            ]
        )
        self.mod = mod

        self.mcfitter = CurveFitter(self.objective)

    def test_bounds_list(self):
        bnds = bounds_list(self.p)
        assert_allclose(bnds, [(-100, 100), (-100, 100)])

        # try making a Parameter bound a normal distribution, then get an
        # approximation to box bounds
        self.p[0].bounds = PDF(norm(0, 1))
        assert_allclose(
            bounds_list(self.p), [norm(0, 1).ppf([0.005, 0.995]), (-100, 100)]
        )

    def test_constraints(self):
        # constraints should work during fitting
        self.p[0].value = 5.4

        self.p[1].constraint = -0.203 * self.p[0]
        assert_equal(self.p[1].value, self.p[0].value * -0.203)
        res = self.mcfitter.fit()

        assert_(res.success)
        assert_equal(len(self.objective.varying_parameters()), 1)

        # lnsigma is parameters[0]
        assert_(self.p[0] is self.objective.parameters.flattened()[0])
        assert_(self.p[1] is self.objective.parameters.flattened()[1])
        assert_almost_equal(self.p[0].value, res.x[0])
        assert_almost_equal(self.p[1].value, self.p[0].value * -0.203)

        # check that constraints work during sampling
        # the CurveFitter has to be set up again if you change how the
        # parameters are being fitted.
        mcfitter = CurveFitter(self.objective)
        assert_(mcfitter.nvary == 1)
        mcfitter.sample(5)
        assert_equal(self.p[1].value, self.p[0].value * -0.203)
        # the constrained parameters should have a chain
        assert_(self.p[0].chain is not None)
        assert_(self.p[1].chain is not None)
        assert_allclose(self.p[1].chain, self.p[0].chain * -0.203)

    def test_mcmc(self):
        self.mcfitter.sample(steps=50, nthin=1, verbose=False)

        assert_equal(self.mcfitter.nvary, 2)

        # smoke test for corner plot
        self.mcfitter.objective.corner()

        # we're not doing Parallel Tempering here.
        assert_(self.mcfitter._ntemps == -1)
        assert_(isinstance(self.mcfitter.sampler, emcee.EnsembleSampler))

        # should be able to multithread
        mcfitter = CurveFitter(self.objective, nwalkers=50)
        res = mcfitter.sample(steps=33, nthin=2, verbose=False, pool=2)

        # check that the autocorrelation function at least runs
        acfs = mcfitter.acf(nburn=10)
        assert_equal(acfs.shape[-1], mcfitter.nvary)

        # check the standalone autocorrelation calculator
        acfs2 = autocorrelation_chain(mcfitter.chain, nburn=10)
        assert_equal(acfs, acfs2)

        # check integrated_time
        integrated_time(acfs2, tol=5)

        # check chain shape
        assert_equal(mcfitter.chain.shape, (33, 50, 2))
        # assert_equal(mcfitter._lastpos, mcfitter.chain[:, -1, :])
        assert_equal(res[0].chain.shape, (33, 50))

        # if the number of parameters changes there should be an Exception
        # raised
        from pytest import raises

        with raises(RuntimeError):
            self.p[0].vary = False
            self.mcfitter.sample(1)

        # can fix by making the sampler again
        self.mcfitter.make_sampler()
        self.mcfitter.sample(1)

    def test_random_seed(self):
        # check that MCMC sampling is reproducible
        self.mcfitter.sample(steps=2, random_state=1)

        # get a starting pos
        starting_pos = self.mcfitter._state.coords

        # is sampling reproducible
        self.mcfitter.reset()
        self.mcfitter.initialise(pos=starting_pos)
        self.mcfitter.sample(3, random_state=1, pool=1)
        chain1 = np.copy(self.mcfitter.chain)

        self.mcfitter.reset()
        self.mcfitter.initialise(pos=starting_pos)
        self.mcfitter.sample(3, random_state=1, pool=1)
        chain2 = np.copy(self.mcfitter.chain)

        assert_equal(chain1, chain2)

    def test_mcmc_pt(self):
        # smoke test for parallel tempering
        x = np.array(self.objective.parameters)

        mcfitter = CurveFitter(self.objective, ntemps=10, nwalkers=50)
        assert_equal(mcfitter.sampler.ntemps, 10)

        assert len(list(flatten(self.objective.parameters))) == 2
        # check that the parallel sampling works
        # and that chain shape is correct
        res = mcfitter.sample(steps=5, nthin=2, verbose=False, pool=-1)
        assert_equal(mcfitter.chain.shape, (5, 10, 50, 2))
        assert_equal(res[0].chain.shape, (5, 50))
        assert_equal(mcfitter.chain[:, 0, :, 0], res[0].chain)
        assert_equal(mcfitter.chain[:, 0, :, 1], res[1].chain)
        chain = np.copy(mcfitter.chain)

        assert len(list(flatten(self.objective.parameters))) == 2

        # the sampler should store the probability
        assert_equal(mcfitter.logpost.shape, (5, 10, 50))
        assert_allclose(mcfitter.logpost, mcfitter.sampler._ptchain.logP)

        logprobs = mcfitter.logpost
        highest_prob_loc = np.argmax(logprobs[:, 0])
        idx = np.unravel_index(highest_prob_loc, logprobs[:, 0].shape)
        idx = list(idx)
        idx.insert(1, 0)
        idx = tuple(idx)
        assert_equal(idx, mcfitter.index_max_prob)
        pvals = mcfitter.chain[idx]
        assert_allclose(logprobs[idx], self.objective.logpost(pvals))

        # try resetting the chain
        mcfitter.reset()

        # test for reproducible operation
        self.objective.setp(x)
        mcfitter = CurveFitter(self.objective, ntemps=10, nwalkers=50)
        mcfitter.initialise("jitter", random_state=1)
        mcfitter.sample(steps=5, nthin=2, verbose=False, random_state=2)
        chain = np.copy(mcfitter.chain)

        self.objective.setp(x)
        mcfitter = CurveFitter(self.objective, ntemps=10, nwalkers=50)
        mcfitter.initialise("jitter", random_state=1)
        mcfitter.sample(steps=5, nthin=2, verbose=False, random_state=2)
        chain2 = np.copy(mcfitter.chain)

        assert_allclose(chain2, chain)

    def test_mcmc_init(self):
        # smoke test for sampler initialisation
        # TODO check that the initialisation worked.
        # reproducible initialisation with random_state dependents
        self.mcfitter.initialise("prior", random_state=1)
        starting_pos = np.copy(self.mcfitter._state.coords)

        self.mcfitter.initialise("prior", random_state=1)
        starting_pos2 = self.mcfitter._state.coords
        assert_equal(starting_pos, starting_pos2)

        self.mcfitter.initialise("jitter", random_state=1)
        starting_pos = np.copy(self.mcfitter._state.coords)

        self.mcfitter.initialise("jitter", random_state=1)
        starting_pos2 = self.mcfitter._state.coords
        assert_equal(starting_pos, starting_pos2)

        mcfitter = CurveFitter(self.objective, nwalkers=100)
        mcfitter.initialise("covar")
        assert_equal(mcfitter._state.coords.shape, (100, 2))
        mcfitter.initialise("prior")
        assert_equal(mcfitter._state.coords.shape, (100, 2))
        mcfitter.initialise("jitter")
        assert_equal(mcfitter._state.coords.shape, (100, 2))
        # initialise with last position
        mcfitter.sample(steps=1)
        chain = mcfitter.chain
        mcfitter.initialise(pos=chain[-1])
        assert_equal(mcfitter._state.coords.shape, (100, 2))
        # initialise with chain
        mcfitter.sample(steps=2)
        chain = mcfitter.chain
        mcfitter.initialise(pos=chain)
        assert_equal(mcfitter._state.coords, chain[-1])
        # initialise with chain if it's never been run before
        mcfitter = CurveFitter(self.objective, nwalkers=100)
        mcfitter.initialise(chain)

        # initialise for Parallel tempering
        mcfitter = CurveFitter(self.objective, ntemps=20, nwalkers=100)
        mcfitter.initialise("covar")
        assert_equal(mcfitter._state.coords.shape, (20, 100, 2))
        mcfitter.initialise("prior")
        assert_equal(mcfitter._state.coords.shape, (20, 100, 2))
        mcfitter.initialise("jitter")
        assert_equal(mcfitter._state.coords.shape, (20, 100, 2))
        # initialise with last position
        mcfitter.sample(steps=1)
        chain = mcfitter.chain
        mcfitter.initialise(pos=chain[-1])
        assert_equal(mcfitter._state.coords.shape, (20, 100, 2))
        # initialise with chain
        mcfitter.sample(steps=2)
        chain = mcfitter.chain
        mcfitter.initialise(pos=np.copy(chain))
        assert_equal(mcfitter._state.coords, chain[-1])
        # initialise with chain if it's never been run before
        mcfitter = CurveFitter(self.objective, nwalkers=100, ntemps=20)
        mcfitter.initialise(chain)

    def test_fit_smoke(self):
        # smoke tests to check that fit runs
        def callback(xk):
            return

        def callback2(xk, **kws):
            return

        # L-BFGS-B
        res0 = self.mcfitter.fit(callback=callback)
        assert_almost_equal(res0.x, [self.b_ls, self.m_ls], 6)
        res0 = self.mcfitter.fit()
        res0 = self.mcfitter.fit(verbose=False)
        res0 = self.mcfitter.fit(verbose=False, callback=callback)

        # least_squares
        res1 = self.mcfitter.fit(method="least_squares")
        assert_almost_equal(res1.x, [self.b_ls, self.m_ls], 6)

        # least_squares doesn't accept a callback. As well as testing that
        # least_squares works, it checks that providing a callback doesn't
        # trip the fitter up.
        res1 = self.mcfitter.fit(method="least_squares", callback=callback)
        assert_almost_equal(res1.x, [self.b_ls, self.m_ls], 6)

        # need full bounds for differential_evolution
        self.p[0].range(3, 7)
        self.p[1].range(-2, 0)
        res2 = self.mcfitter.fit(
            method="differential_evolution",
            seed=1,
            popsize=10,
            maxiter=100,
            callback=callback2,
        )
        assert_almost_equal(res2.x, [self.b_ls, self.m_ls], 6)

        # check that the res object has covar and stderr
        assert_("covar" in res0)
        assert_("stderr" in res0)

    def test_NIST(self):
        # Run all the NIST standard tests with leastsq
        for model in NIST_Models:
            try:
                NIST_runner(model)
            except Exception:
                print(model)
                raise


def gauss(x, p0):
    p = np.array(p0)
    return p[0] + p[1] * np.exp(-(((x - p[2]) / p[3]) ** 2))


class TestFitterGauss:
    # Test CurveFitter with a noisy gaussian, weighted and unweighted, to see
    # if the parameters and uncertainties come out correct

    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.tmpdir = tmpdir.strpath

        theoretical = np.loadtxt(os.path.join(self.path, "gauss_data.txt"))
        xvals, yvals, evals = np.hsplit(theoretical, 3)
        xvals = xvals.flatten()
        yvals = yvals.flatten()
        evals = evals.flatten()

        # these best weighted values and uncertainties obtained with Igor
        self.best_weighted = [-0.00246095, 19.5299, -8.28446e-2, 1.24692]

        self.best_weighted_errors = [
            0.0220313708486,
            1.12879436221,
            0.0447659158681,
            0.0412022938883,
        ]

        self.best_weighted_chisqr = 77.6040960351

        self.best_unweighted = [
            -0.10584111872702096,
            19.240347049328989,
            0.0092623066070940396,
            1.501362314145845,
        ]

        self.best_unweighted_errors = [
            0.34246565477,
            0.689820935208,
            0.0411243173041,
            0.0693429375282,
        ]

        self.best_unweighted_chisqr = 497.102084956

        self.p0 = np.array([0.1, 20.0, 0.1, 0.1])
        self.names = ["bkg", "A", "x0", "width"]
        self.bounds = [(-1, 1), (0, 30), (-5.0, 5.0), (0.001, 2)]

        self.params = Parameters(name="gauss_params")
        for p, name, bound in zip(self.p0, self.names, self.bounds):
            param = Parameter(p, name=name)
            param.range(*bound)
            param.vary = True
            self.params.append(param)

        self.model = Model(self.params, fitfunc=gauss)
        self.data = Data1D((xvals, yvals, evals))
        self.objective = Objective(self.model, self.data)
        return 0

    def test_pickle(self):
        # tests if a CurveFitter can be pickled/unpickled.
        f = CurveFitter(self.objective)
        pkl = pickle.dumps(f)
        g = pickle.loads(pkl)
        g._check_vars_unchanged()

    def test_best_weighted(self):
        assert_equal(len(self.objective.varying_parameters()), 4)
        self.objective.setp(self.p0)

        f = CurveFitter(self.objective, nwalkers=100)
        res = f.fit("least_squares", jac="3-point")

        output = res.x
        assert_almost_equal(output, self.best_weighted, 3)
        assert_almost_equal(
            self.objective.chisqr(), self.best_weighted_chisqr, 5
        )

        # compare the residuals
        res = (self.data.y - self.model(self.data.x)) / self.data.y_err
        assert_equal(self.objective.residuals(), res)

        # compare objective.covar to the best_weighted_errors
        uncertainties = [param.stderr for param in self.params]
        assert_allclose(uncertainties, self.best_weighted_errors, rtol=0.005)

        # we're also going to try the checkpointing here.
        checkpoint = os.path.join(self.tmpdir, "checkpoint.txt")

        # compare samples to best_weighted_errors
        np.random.seed(1)
        f.sample(steps=201, random_state=1, verbose=False, f=checkpoint)
        process_chain(self.objective, f.chain, nburn=50, nthin=10)
        uncertainties = [param.stderr for param in self.params]
        assert_allclose(uncertainties, self.best_weighted_errors, rtol=0.07)

        # test that the checkpoint worked
        check_array = np.loadtxt(checkpoint)
        check_array = check_array.reshape(201, f._nwalkers, f.nvary)
        assert_allclose(check_array, f.chain)

        # test loading the checkpoint
        chain = load_chain(checkpoint)
        assert_allclose(chain, f.chain)

        f.initialise("jitter")
        f.sample(steps=2, nthin=4, f=checkpoint, verbose=False)
        assert_equal(f.chain.shape[0], 2)

        # we should be able to produce 2 * 100 steps from the generator
        g = self.objective.pgen(ngen=20000000000)
        s = [i for i, a in enumerate(g)]
        assert_equal(np.max(s), 200 - 1)
        g = self.objective.pgen(ngen=200)
        pvec = next(g)
        assert_equal(pvec.size, len(self.objective.parameters.flattened()))

        # check that all the parameters are returned via pgen, not only those
        # being varied.
        self.params[0].vary = False
        f = CurveFitter(self.objective, nwalkers=100)
        f.initialise("jitter")
        f.sample(steps=2, nthin=4, f=checkpoint, verbose=False)
        g = self.objective.pgen(ngen=100)
        pvec = next(g)
        assert_equal(pvec.size, len(self.objective.parameters.flattened()))

        # the following test won't work because of emcee/gh226.
        # chain = load_chain(checkpoint)
        # assert_(chain.shape == f.chain.shape)
        # assert_allclose(chain, f.chain)

        # try reproducing best fit with parallel tempering
        self.params[0].vary = True
        f = CurveFitter(self.objective, nwalkers=100, ntemps=10)
        f.fit("differential_evolution", seed=1)

        f.sample(steps=201, random_state=1, verbose=False)
        process_chain(self.objective, f.chain, nburn=50, nthin=15)
        print(self.params[0].chain.shape, self.params[0].chain)

        uncertainties = [param.stderr for param in self.params]
        assert_allclose(uncertainties, self.best_weighted_errors, rtol=0.07)

    def test_best_unweighted(self):
        self.objective.weighted = False
        f = CurveFitter(self.objective, nwalkers=100)
        res = f.fit()

        output = res.x
        assert_almost_equal(
            self.objective.chisqr(), self.best_unweighted_chisqr
        )
        assert_almost_equal(output, self.best_unweighted, 5)

        # compare the residuals
        res = self.data.y - self.model(self.data.x)
        assert_equal(self.objective.residuals(), res)

        # compare objective._covar to the best_unweighted_errors
        uncertainties = np.array([param.stderr for param in self.params])
        assert_almost_equal(uncertainties, self.best_unweighted_errors, 3)

        # the samples won't compare to the covariance matrix...
        # f.sample(nsteps=150, nburn=20, nthin=30, random_state=1)
        # uncertainties = [param.stderr for param in self.params]
        # assert_allclose(uncertainties, self.best_unweighted_errors,
        #                 rtol=0.15)

    def test_all_minimisers(self):
        """test minimisers against the Gaussian fit"""
        f = CurveFitter(self.objective)

        methods = [
            "differential_evolution",
            "L-BFGS-B",
            "least_squares",
            "shgo",
            "dual_annealing",
        ]

        for method in methods:
            self.objective.setp(self.p0)
            opts = {}
            if method in ["differential_evolution", "dual_annealing"]:
                opts = {"seed": 1}

            res = f.fit(method=method, **opts)
            assert_allclose(res.x, self.best_weighted, rtol=0.005)

        # smoke test to check that we can use nlpost
        self.objective.setp(self.p0)
        logp0 = self.objective.logp()

        # check that probabilities are calculated correctly
        assert_allclose(
            self.objective.logpost(),
            self.objective.logp() + self.objective.logl(),
        )
        assert_allclose(self.objective.nlpost(), -self.objective.logpost())
        assert_allclose(
            self.objective.nlpost(self.p0), -self.objective.logpost(self.p0)
        )

        # if the priors are all uniform then the only difference between
        # logpost and logl is a constant. A minimiser should converge on the
        # same answer. The following tests examine that.
        # The test works for dual_annealing, but not for differential
        # evolution, not sure why that is.
        self.objective.setp(self.p0)
        res1 = f.fit(method="dual_annealing", seed=1)
        assert_almost_equal(res1.x, self.best_weighted, 3)
        nll1 = self.objective.nll()
        nlpost1 = self.objective.nlpost()

        self.objective.setp(self.p0)
        res2 = f.fit(method="dual_annealing", target="nlpost", seed=1)
        assert_almost_equal(res2.x, self.best_weighted, 3)
        nll2 = self.objective.nll()
        nlpost2 = self.objective.nlpost()

        assert_allclose(nlpost1, nlpost2, atol=0.001)
        assert_allclose(nll1, nll2, atol=0.001)

        # these two priors are calculated for different parameter values
        # (before and after the fit) they should be the same because all
        # the parameters have uniform priors.
        assert_almost_equal(self.objective.logp(), logp0)

    def test_pymc3_sample(self):
        # test sampling with pymc3
        try:
            import pymc3 as pm
            from refnx.analysis import pymc3_model
        except (ModuleNotFoundError, ImportError, AttributeError):
            # can't run test if pymc3/theano not installed
            return

        with pymc3_model(self.objective):
            s = pm.NUTS()
            pm.sample(
                200,
                tune=100,
                step=s,
                discard_tuned_samples=True,
                compute_convergence_checks=False,
                random_seed=1,
            )
            # don't check for accuracy because it requires a heap more
            # draws.
            # means = [np.mean(trace[f"p{i}"]) for i in range(4)]
            # assert_allclose(means, self.best_weighted, rtol=0.04)
            # errors = [np.std(trace[f"p{i}"]) for i in range(4)]
            # assert_allclose(errors, self.best_weighted_errors, atol=0.02)


"""
        The Gaussian example sampling can also be performed with pymc3.
        The above results from emcee have been verified against pymc3 - the
        unweighted sampling statistics are the same.

        from pymc3 import (Model, Normal, HalfNormal, Flat, Uniform,
                           find_MAP, NUTS, sample, summary, traceplot)

        basic_model = Model()

        with basic_model:
            # Priors for unknown model parameters
            bkg = Uniform('bkg', -1, 5)
            A0 = Uniform('A0', 0, 50)
            x0 = Uniform('x0', min(x), max(x))
            width = Uniform('width', 0.5, 10)

            # Expected value of outcome
            mu = bkg + A0 * np.exp(-((x - x0) / width) ** 2)

            # Likelihood (sampling distribution) of observations
            #     y_obs = Normal('y_obs', mu=mu, sd=e, observed=y)
            y_obs = Normal('y_obs', mu=mu, observed=y)

        with basic_model:
            # draw 500 posterior samples
            trace = sample(500)
        summary(trace)
"""
