"""
This module tests the objective function by comparing it to the line example
from http://dan.iel.fm/emcee/current/user/line/
"""
import unittest
import pickle

import emcee
from scipy.optimize import minimize
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_)

from refnx.analysis import (Parameter, Model, Objective, BaseObjective,
                            Transform)
from refnx.dataset import Data1D


def line(x, params, *args, **kwds):
    p_arr = np.array(params)
    return p_arr[0] + x * p_arr[1]


class TestObjective(unittest.TestCase):

    def setUp(self):
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

        self.p = Parameter(self.b_ls, 'b') | Parameter(self.m_ls, 'm')
        self.model = Model(self.p, fitfunc=line)
        self.objective = Objective(self.model, self.data,
                                   lnsigma=Parameter(0, 'lnsigma', vary=True))

        # want b and m and lnsigma to vary
        self.p[0].vary = True
        self.p[1].vary = True
        self.objective.lnsigma.vary = True

        assert_equal(self.objective.lnsigma.value, 0)
        mod = np.array([4.78166609, 4.42364699, 4.16404064, 3.50343504,
                        3.4257084, 2.93594347, 2.92035638, 2.67533842,
                        2.28136038, 2.19772983, 1.99295496, 1.93748334,
                        1.87484436, 1.65161016, 1.44613461, 1.11128101,
                        1.04584535, 0.86055984, 0.76913963, 0.73906649,
                        0.73331407, 0.68350418, 0.65216599, 0.59838566,
                        0.13070299, 0.10749131, -0.01010195, -0.10010155,
                        -0.29495372, -0.42817431, -0.43122391, -0.64637715,
                        -1.30560686, -1.32626428, -1.44835768, -1.52589881,
                        -1.56371158, -2.12048349, -2.24899179, -2.50292682,
                        -2.53576659, -2.55797996, -2.60870542, -2.7074727,
                        -3.93781479, -4.12415366, -4.42313742, -4.98368609,
                        -5.38782395, -5.44077086])
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
        self.objective.lnsigma.vary = False

        assert_(len(self.objective.varying_parameters()) == 1)
        self.objective.setp(np.array([1.23]))
        assert_equal(self.p[1].value, 1.23)
        self.objective.setp(np.array([0, 1.234, 1.23]))
        assert_equal(np.array(self.p), [1.234, 1.23])

    def test_pvals(self):
        assert_equal(self.objective.parameters.pvals,
                     [0, self.b_ls, self.m_ls])
        self.objective.parameters.pvals = [0, 1, 2]
        assert_equal(self.objective.parameters.pvals, [0, 1, 2.])

    def test_lnprior(self):
        self.objective.lnsigma = 0

        self.p[0].range(0, 10)
        assert_almost_equal(self.objective.lnprior(), np.log(0.1))

        # lnprior should set parameters
        self.objective.lnprior([8, 2])
        assert_equal(np.array(self.objective.parameters), [8, 2])

        # if we supply a value outside the range it should return -inf
        assert_equal(self.objective.lnprior([-1, 2]), -np.inf)

    def test_lnprob(self):
        # http://dan.iel.fm/emcee/current/user/line/
        assert_almost_equal(self.objective.lnprior(), 0)
        # the uncertainties are underestimated in this example...
        assert_almost_equal(self.objective.lnlike(), -559.01078135444595)
        assert_almost_equal(self.objective.lnprob(), -559.01078135444595)

    def test_chisqr(self):
        assert_almost_equal(self.objective.chisqr(),
                            1231.1096772954229)

    def test_residuals(self):
        assert_almost_equal(self.objective.residuals(),
                            (self.data.y - self.mod) / self.data.y_err)

    def test_objective_pickle(self):
        # can you pickle the objective function?
        pkl = pickle.dumps(self.objective)
        pickle.loads(pkl)

    def test_transform_pickle(self):
        # can you pickle the Transform object?
        pkl = pickle.dumps(Transform('logY'))
        pickle.loads(pkl)

    def test_base_emcee(self):
        # check that the base objective works against the emcee example.
        def lnprior(theta, x, y, yerr):
            m, b, lnf = theta
            if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
                return 0.0
            return -np.inf

        def lnlike(theta, x, y, yerr):
            m, b, lnf = theta
            model = m * x + b
            inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
            return -0.5 * (np.sum((y - model)**2 * inv_sigma2 -
                                  np.log(inv_sigma2)))

        x, y, yerr, _ = self.data.data

        theta = [self.m_true, self.b_true, np.log(self.f_true)]
        bo = BaseObjective(theta, lnlike, lnprior=lnprior,
                           fcn_args=(x, y, yerr))

        # test that the wrapper gives the same lnlike as the direct function
        assert_almost_equal(bo.lnlike(theta),
                            lnlike(theta, x, y, yerr))
        assert_almost_equal(bo.lnlike(theta), -bo.nll(theta))
        assert_almost_equal(bo.nll(theta), 12.8885352412)

        # Find the maximum likelihood value.
        result = minimize(bo.nll, theta)

        ndim, nwalkers = 3, 100
        pos = [result["x"] + 1e-4 * np.random.randn(ndim) for
               i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, bo.lnprob)
        sampler.run_mcmc(pos, 500, rstate0=np.random.get_state())

        burnin = 50
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        samples[:, 2] = np.exp(samples[:, 2])
        m_mc, b_mc, f_mc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                               zip(*np.percentile(samples, [16, 50, 84],
                                                  axis=0)))
        assert_almost_equal(m_mc, (-1.006610168076133,
                                   0.076119420801836424,
                                   0.077341488854482332))

        assert_almost_equal(b_mc, (4.5416267081353094,
                                   0.35581750201588402,
                                   0.34489351756496678))

        assert_almost_equal(f_mc, (0.46264188436968745,
                                   0.078955744624578661,
                                   0.061168332596617969))

        # # smoke test for covariance matrix
        bo.parameters = np.array(result['x'])
        covar1 = bo.covar()
        uncertainties = np.sqrt(np.diag(covar1))

        # covariance from objective._covar should be almost equal to
        # the covariance matrix from sampling
        covar2 = np.cov(samples.T)
        assert_almost_equal(np.sqrt(np.diag(covar2))[:2],
                            uncertainties[:2],
                            2)

        # check covariance of self.objective
        # TODO
        var_arr = result['x'][:]
        var_arr[0], var_arr[1], var_arr[2] = var_arr[2], var_arr[1], var_arr[0]

        assert_(self.objective.data.weighted)
        self.objective.parameters.pvals = var_arr

        # covar3 = self.objective.covar()
        # uncertainties3 = np.sqrt(np.diag(covar3))
        # assert_almost_equal(uncertainties3, uncertainties)
        # assert(False)
