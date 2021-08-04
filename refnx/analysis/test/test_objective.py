"""
This module tests the objective function by comparing it to the line example
from http://dan.iel.fm/emcee/current/user/line/
"""
import pickle
from multiprocessing.reduction import ForkingPickler
import os

import pytest

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import minimize, least_squares
from scipy.optimize._numdiff import approx_derivative
import scipy.stats as stats

from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_,
    assert_allclose,
)

from refnx.analysis import (
    Parameter,
    Model,
    Objective,
    BaseObjective,
    Transform,
    Parameters,
    PDF,
)
from refnx.dataset import Data1D, ReflectDataset
from refnx.util import ErrorProp as EP
from refnx._lib import emcee


def line(x, params, *args, **kwds):
    p_arr = np.array(params)
    return p_arr[0] + x * p_arr[1]


def gauss(x, p0):
    p = np.array(p0)
    return p[0] + p[1] * np.exp(-(((x - p[2]) / p[3]) ** 2))


def logp_extra(model, data):
    return 1.0


def line_ND(x, params):
    # a fit function for testing multidimensionality
    p_arr = np.array(params)
    y0 = p_arr[0] + x[:, 0] * p_arr[1]
    y1 = p_arr[0] + x[:, 1] * p_arr[1]
    return np.c_[y0, y1]


class TestObjective:
    def setup_method(self):
        # Choose the "true" parameters.

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

        self.p = Parameter(self.b_ls, "b") | Parameter(self.m_ls, "m")
        self.model = Model(self.p, fitfunc=line)
        self.objective = Objective(self.model, self.data)

        # want b and m
        self.p[0].vary = True
        self.p[1].vary = True

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

    def test_model(self):
        # test that the line data produced by our model is the same as the
        # test data
        assert_almost_equal(self.model(self.data.x), self.mod)

    def test_synthetic_data(self):
        # test that we create the correct synthetic data by performing a least
        # squares fit on it
        assert_(self.data.y_err is not None)

        x, y, y_err, _ = self.data.data
        A = np.vstack((np.ones_like(x), x)).T
        C = np.diag(y_err * y_err)
        cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
        b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

        assert_almost_equal(b_ls, self.b_ls)
        assert_almost_equal(m_ls, self.m_ls)

    def test_setp(self):
        # check that we can set parameters
        self.p[0].vary = False

        assert_(len(self.objective.varying_parameters()) == 1)
        self.objective.setp(np.array([1.23]))
        assert_equal(self.p[1].value, 1.23)
        self.objective.setp(np.array([1.234, 1.23]))
        assert_equal(np.array(self.p), [1.234, 1.23])

    def test_pvals(self):
        assert_equal(self.objective.parameters.pvals, [self.b_ls, self.m_ls])
        self.objective.parameters.pvals = [1, 2]
        assert_equal(self.objective.parameters.pvals, [1, 2.0])

    def test_logp(self):
        self.p[0].range(0, 10)
        assert_almost_equal(self.objective.logp(), np.log(0.1))

        # logp should set parameters
        self.objective.logp([8, 2])
        assert_equal(np.array(self.objective.parameters), [8, 2])

        # if we supply a value outside the range it should return -inf
        assert_equal(self.objective.logp([-1, 2]), -np.inf)

        # are auxiliary parameters included in log?
        assert_almost_equal(self.objective.logp([8, 2]), np.log(0.1))
        p = Parameter(2.0, bounds=(1.0, 3.0))
        self.objective.auxiliary_params = Parameters([p])
        assert len(self.objective.varying_parameters()) == 2
        assert_equal(self.objective.logp(), np.log(0.1))
        assert p in self.objective.parameters.flattened()
        p.vary = True
        assert len(self.objective.varying_parameters()) == 3
        assert_equal(self.objective.logp(), np.log(0.1) + np.log(0.5))
        assert p in self.objective.varying_parameters().flattened()

    def test_logpost(self):
        # http://dan.iel.fm/emcee/current/user/line/
        assert_almost_equal(self.objective.logp(), 0)

        assert_almost_equal(self.objective.nlpost(), -self.objective.logpost())

        # the uncertainties are underestimated in this example...
        # amendment factor because dfm emcee example does not include 2pi
        amend = 0.5 * self.objective.npoints * np.log(2 * np.pi)
        assert_almost_equal(self.objective.logl() + amend, -559.01078135444595)
        assert_almost_equal(
            self.objective.logpost() + amend, -559.01078135444595
        )

    def test_prior_transform(self):
        self.p[0].bounds = PDF(stats.uniform(-10, 20))
        self.p[1].bounds = PDF(stats.norm(loc=5, scale=10))
        x = self.objective.prior_transform([0.1, 0.9])
        assert_allclose(
            x,
            stats.uniform.ppf(0.1, -10, 20),
            stats.norm.ppf(0.9, loc=5, scale=10),
        )

    def test_chisqr(self):
        assert_almost_equal(self.objective.chisqr(), 1231.1096772954229)

    def test_residuals(self):
        # weighted, with and without transform
        assert_almost_equal(
            self.objective.residuals(),
            (self.data.y - self.mod) / self.data.y_err,
        )

        objective = Objective(
            self.model, self.data, transform=Transform("lin")
        )
        assert_almost_equal(
            objective.residuals(), (self.data.y - self.mod) / self.data.y_err
        )

        # unweighted, with and without transform
        objective = Objective(self.model, self.data, use_weights=False)
        assert_almost_equal(objective.residuals(), self.data.y - self.mod)

        objective = Objective(
            self.model,
            self.data,
            use_weights=False,
            transform=Transform("lin"),
        )
        assert_almost_equal(objective.residuals(), self.data.y - self.mod)

    def test_masked_dataset(self):
        residuals = self.objective.residuals()

        mask = np.full_like(self.objective.data.y, True, bool)
        mask[1] = False
        self.objective.data.mask = mask

        assert_equal(self.objective.residuals().size, residuals.size - 1)

    def test_logp_extra(self):
        original_logl = self.objective.logl()
        self.objective.logp_extra = logp_extra
        assert_almost_equal(self.objective.logl(), original_logl + 1)

    def test_objective_pickle(self):
        # can you pickle the objective function?
        pkl = pickle.dumps(self.objective)
        pickle.loads(pkl)

        # check the ForkingPickler as well.
        if hasattr(ForkingPickler, "dumps"):
            pkl = ForkingPickler.dumps(self.objective)
            pickle.loads(pkl)

        # can you pickle with an extra function present?
        self.objective.logp_extra = logp_extra
        pkl = pickle.dumps(self.objective)
        pickle.loads(pkl)

        # check the ForkingPickler as well.
        if hasattr(ForkingPickler, "dumps"):
            pkl = ForkingPickler.dumps(self.objective)
            pickle.loads(pkl)

    def test_transform_pickle(self):
        # can you pickle the Transform object?
        pkl = pickle.dumps(Transform("logY"))
        pickle.loads(pkl)

    def test_transform(self):
        pth = os.path.dirname(os.path.abspath(__file__))

        fname = os.path.join(pth, "c_PLP0011859_q.txt")
        data = ReflectDataset(fname)
        t = Transform("logY")

        yt, et = t(data.x, data.y, y_err=data.y_err)
        assert_equal(yt, np.log10(data.y))

        yt, _ = t(data.x, data.y, y_err=None)
        assert_equal(yt, np.log10(data.y))

        EPy, EPe = EP.EPlog10(data.y, data.y_err)
        assert_equal(yt, EPy)
        assert_equal(et, EPe)

    def test_repr_transform(self):
        p = Transform(None)
        q = eval(repr(p))
        assert p.form == q.form

        p = Transform("logY")
        q = eval(repr(p))
        assert p.form == q.form

    def test_lnsigma(self):
        # check that lnsigma works correctly, by using the emcee line fit
        # example
        def logp(theta, x, y, yerr):
            m, b, lnf = theta
            if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
                return 0.0
            return -np.inf

        def logl(theta, x, y, yerr):
            m, b, lnf = theta
            model = m * x + b
            inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
            print(inv_sigma2)
            return -0.5 * (
                np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2))
            )

        x, y, yerr, _ = self.data.data

        theta = [self.m_true, self.b_true, np.log(self.f_true)]
        bo = BaseObjective(theta, logl, logp=logp, fcn_args=(x, y, yerr))

        lnsigma = Parameter(
            np.log(self.f_true), "lnsigma", bounds=(-10, 1), vary=True
        )
        self.objective.setp(np.array([self.b_true, self.m_true]))
        self.objective.lnsigma = lnsigma

        # amendment factor because dfm emcee example does not include 2pi
        amend = 0.5 * self.objective.npoints * np.log(2 * np.pi)

        assert_allclose(self.objective.logl() + amend, bo.logl())

    def test_base_emcee(self):
        # check that the base objective works against the emcee example.
        def logp(theta, x, y, yerr):
            m, b, lnf = theta
            if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
                return 0.0
            return -np.inf

        def logl(theta, x, y, yerr):
            m, b, lnf = theta
            model = m * x + b
            inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
            return -0.5 * (
                np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2))
            )

        x, y, yerr, _ = self.data.data

        theta = [self.m_true, self.b_true, np.log(self.f_true)]
        bo = BaseObjective(theta, logl, logp=logp, fcn_args=(x, y, yerr))

        # test that the wrapper gives the same logl as the direct function
        assert_almost_equal(bo.logl(theta), logl(theta, x, y, yerr))
        assert_almost_equal(bo.logl(theta), -bo.nll(theta))
        assert_almost_equal(bo.nll(theta), 12.8885352412)

        # Find the maximum likelihood value.
        result = minimize(bo.nll, theta)

        # for repeatable sampling
        np.random.seed(1)

        ndim, nwalkers = 3, 100
        pos = [
            result["x"] + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)
        ]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, bo.logpost)
        state = emcee.State(pos, random_state=np.random.get_state())
        sampler.run_mcmc(state, 800)

        burnin = 200
        samples = sampler.get_chain()[burnin:, :, :].reshape((-1, ndim))
        samples[:, 2] = np.exp(samples[:, 2])
        m_mc, b_mc, f_mc = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(samples, [16, 50, 84], axis=0)),
        )
        assert_allclose(m_mc, (-1.0071664, 0.0809444, 0.0784894), rtol=0.04)

        assert_allclose(b_mc, (4.5428107, 0.3549174, 0.3673304), rtol=0.04)

        assert_allclose(f_mc, (0.4610898, 0.0823304, 0.0640812), rtol=0.06)

        # # smoke test for covariance matrix
        bo.parameters = np.array(result["x"])
        covar1 = bo.covar()
        uncertainties = np.sqrt(np.diag(covar1))

        # covariance from objective._covar should be almost equal to
        # the covariance matrix from sampling
        covar2 = np.cov(samples.T)
        assert_almost_equal(np.sqrt(np.diag(covar2))[:2], uncertainties[:2], 2)

        # check covariance of self.objective
        # TODO
        var_arr = result["x"][:]
        var_arr[0], var_arr[1], var_arr[2] = var_arr[2], var_arr[1], var_arr[0]

        # assert_(self.objective.data.weighted)
        # self.objective.parameters.pvals = var_arr
        # covar3 = self.objective.covar()
        # uncertainties3 = np.sqrt(np.diag(covar3))
        # assert_almost_equal(uncertainties3, uncertainties)
        # assert(False)

    def test_covar(self):
        # checks objective.covar against optimize.least_squares covariance.
        path = os.path.dirname(os.path.abspath(__file__))

        theoretical = np.loadtxt(os.path.join(path, "gauss_data.txt"))
        xvals, yvals, evals = np.hsplit(theoretical, 3)
        xvals = xvals.flatten()
        yvals = yvals.flatten()
        evals = evals.flatten()

        p0 = np.array([0.1, 20.0, 0.1, 0.1])
        names = ["bkg", "A", "x0", "width"]
        bounds = [(-1, 1), (0, 30), (-5.0, 5.0), (0.001, 2)]

        params = Parameters(name="gauss_params")
        for p, name, bound in zip(p0, names, bounds):
            param = Parameter(p, name=name)
            param.range(*bound)
            param.vary = True
            params.append(param)

        model = Model(params, fitfunc=gauss)
        data = Data1D((xvals, yvals, evals))
        objective = Objective(model, data)

        # first calculate least_squares jac/hess/covariance matrices
        res = least_squares(
            objective.residuals, np.array(params), jac="3-point"
        )

        hess_least_squares = np.matmul(res.jac.T, res.jac)
        covar_least_squares = np.linalg.inv(hess_least_squares)

        # now calculate corresponding matrices by hand, to see if the approach
        # concurs with least_squares
        objective.setp(res.x)
        _pvals = np.array(res.x)

        def residuals_scaler(vals):
            return np.squeeze(objective.residuals(_pvals * vals))

        jac = approx_derivative(residuals_scaler, np.ones_like(_pvals))
        hess = np.matmul(jac.T, jac)
        covar = np.linalg.inv(hess)

        covar = covar * np.atleast_2d(_pvals) * np.atleast_2d(_pvals).T

        assert_allclose(covar, covar_least_squares)

        # check that objective.covar corresponds to the least_squares
        # covariance matrix, J.T x J
        objective.setp(res.x)
        covar_objective = objective.covar()
        assert_allclose(covar_objective, covar_least_squares)

        # sometimes the residuals method may not be usable, see if
        # objective.covar calculated from a scalar works
        objective.setp(res.x)
        covar_objective = objective.covar("nll")
        assert_allclose(
            np.sqrt(np.diag(covar_objective)),
            np.sqrt(np.diag(covar_least_squares)),
            rtol=0.08,
        )

        # now see what happens with a parameter that has no effect on residuals
        param = Parameter(1.234, name="dummy")
        param.vary = True
        params.append(param)

        from pytest import raises

        with raises(LinAlgError):
            objective.covar()

    @pytest.mark.xfail
    def test_pymc3(self):
        # test objective logl against pymc3

        # don't run this test if pymc3 is not installed
        try:
            import pymc3 as pm
        except ImportError:
            return

        logl = self.objective.logl()

        from refnx.analysis import pymc3_model
        from refnx.analysis.objective import _to_pymc3_distribution

        mod = pymc3_model(self.objective)
        with mod:
            pymc_logl = mod.logp(
                {"p0": self.p[0].value, "p1": self.p[1].value}
            )

        assert_allclose(logl, pymc_logl)

        # now check some of the distributions
        with pm.Model():
            p = Parameter(1, bounds=(1, 10))
            d = _to_pymc3_distribution("a", p)
            assert_almost_equal(d.distribution.logp(2).eval(), p.logp(2))
            assert_(np.isneginf(d.distribution.logp(-1).eval()))

            q = Parameter(1, bounds=PDF(stats.uniform(1, 9)))
            d = _to_pymc3_distribution("b", q)
            assert_almost_equal(d.distribution.logp(2).eval(), q.logp(2))
            assert_(np.isneginf(d.distribution.logp(-1).eval()))

            p = Parameter(1, bounds=PDF(stats.uniform))
            d = _to_pymc3_distribution("c", p)
            assert_almost_equal(d.distribution.logp(0.5).eval(), p.logp(0.5))

            p = Parameter(1, bounds=PDF(stats.norm))
            d = _to_pymc3_distribution("d", p)
            assert_almost_equal(d.distribution.logp(2).eval(), p.logp(2))

            p = Parameter(1, bounds=PDF(stats.norm(1, 10)))
            d = _to_pymc3_distribution("e", p)
            assert_almost_equal(d.distribution.logp(2).eval(), p.logp(2))

    def test_multidimensionality(self):
        # Check that ND data can be used with an objective/model/data
        # (or at least it doesn't stand in the way)
        rng = np.random.default_rng()
        x = rng.uniform(size=100).reshape(50, 2)

        desired = line_ND(x, self.p)
        assert desired.shape == (50, 2)
        data = Data1D((x, desired))

        model = Model(self.p, fitfunc=line_ND)
        y = model(x)
        assert_allclose(y, desired)

        objective = Objective(model, data)
        assert_allclose(objective.chisqr(), 0)
        assert_allclose(objective.generative(), desired)
        assert_allclose(objective.residuals(), 0)
        assert objective.residuals().shape == (50, 2)

        objective.logl()
        objective.logpost()
        covar = objective.covar()
        assert covar.shape == (2, 2)
