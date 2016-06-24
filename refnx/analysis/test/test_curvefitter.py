import os.path
import unittest
import time
from copy import deepcopy

import refnx.analysis.curvefitter as curvefitter
from refnx.analysis.curvefitter import (values, CurveFitter, HAS_EMCEE,
                                        FitFunction,
                                        _parallel_likelihood_calculator)
import numpy as np
from lmfit.minimizer import MinimizerResult
from NISTModels import NIST_runner, Models, ReadNistData

from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_allclose)
SEED = 1


path = os.path.dirname(os.path.abspath(__file__))


def gauss(x, p0, *args):
    p = list(p0.valuesdict().values())

    return p[0] + p[1] * np.exp(-((x - p[2]) / p[3])**2)


class TestFitter(unittest.TestCase):

    def setUp(self):
        self.xdata = np.linspace(-4, 4, 100)
        self.p0 = np.array([0., 1., 0.0, 1.])
        self.bounds = [(-1, 1), (0, 2), (-1, 1.), (0.001, 2)]

        self.params = curvefitter.to_parameters(self.p0 + 0.2, bounds=self.bounds)
        self.final_params = curvefitter.to_parameters(self.p0, bounds=self.bounds)

        self.ydata = gauss(self.xdata, self.final_params)
        self.f = CurveFitter(gauss, (self.xdata, self.ydata), self.params)

    def pvals(self, params):
        return np.asfarray(list(params.valuesdict().values()))

    def test_fitting(self):
        # the simplest test - a really simple gauss curve with perfect data
        res = self.f.fit()
        assert_almost_equal(self.pvals(res.params), self.p0)
        assert_almost_equal(res.chisqr, 0)

    def test_NIST(self):
        # Run all the NIST standard tests with leastsq
        for model in Models.keys():
            try:
                NIST_runner(model)
            except Exception:
                print(model)
                raise

    def test_model_returns_function(self):
        ydata = gauss(self.xdata, self.final_params)
        model = self.f.model(self.final_params)
        assert_almost_equal(ydata, model)

    def test_residuals(self):
        resid = self.f.residuals(self.final_params)
        assert_almost_equal(np.sum(resid**2), 0)

    def test_cost(self):
        resid = self.f.residuals(self.final_params)
        assert_almost_equal(0, np.sum(resid**2))

    def test_leastsq(self):
        # test that a custom method can be used with scipy.optimize.minimize
        res = self.f.fit()
        assert_almost_equal(self.pvals(res.params), self.p0)

    def test_resid_length(self):
        # the residuals length should be equal to the data length
        resid = self.f.residuals(self.params)
        assert_equal(resid.size, self.f.ydata.size)

    def test_scalar_minimize(self):
        assert_equal(self.pvals(self.params), self.p0 + 0.2)
        res = self.f.fit(method='differential_evolution')
        assert_almost_equal(self.pvals(res.params), self.p0, 3)

    def test_holding_parameter(self):
        # holding parameters means that those parameters shouldn't change
        # during a fit
        self.params['p0'].vary = False
        res = self.f.fit()
        assert_almost_equal(self.p0[0] + 0.2, self.params['p0'].value)

    def test_fit_returns_MinimizerResult(self):
        self.params['p0'].vary = False
        res = self.f.fit()
        assert_(isinstance(res, MinimizerResult))

    def test_costfun(self):
        # test user defined costfun
        res = self.f.fit('nelder')

        def costfun(params, generative, y, e):
            return np.sum((y - generative / e) ** 2)

        g = CurveFitter(gauss, (self.xdata, self.ydata), self.params, costfun=costfun)
        res2 = g.fit('nelder')
        assert_almost_equal(self.pvals(res.params), self.pvals(res2.params))

    # def test_emcee_NIST(self):
    #     datasets = ['DanWood']
    #
    #     for dataset in datasets:
    #         NIST_dataset = ReadNistData(dataset)
    #
    #         x, y = (NIST_dataset['x'], NIST_dataset['y'])
    #
    #         params = NIST_dataset['start']
    #
    #         fitfunc = Models[dataset][0]
    #         fitter = CurveFitter(fitfunc, (x, y), params)
    #         res = fitter.emcee(params=params, steps=1500, nwalkers=100,
    #                            burn=600, thin=25, workers=4,
    #                            is_weighted=False, seed=1)
    #         res.params.pop('__lnsigma')
    #         errs = np.array([res.params[par].stderr for par in res.params])
    #         assert_allclose(values(res.params),
    #                         NIST_dataset['cert_values'],
    #                         rtol=1e-2)
    #         # assert_allclose(errs,
    #         #                 NIST_dataset['cert_stderr'],
    #         #                 rtol=0.1)


class TestFitterGauss(unittest.TestCase):
    # Test CurveFitter with a noisy gaussian, weighted and unweighted, to see
    # if the parameters and uncertainties come out correct

    def setUp(self):
        theoretical = np.loadtxt(os.path.join(path, 'gauss_data.txt'))
        xvals, yvals, evals = np.hsplit(theoretical, 3)
        self.xvals = xvals.flatten()
        self.yvals = yvals.flatten()
        self.evals = evals.flatten()

        self.best_weighted = [-0.00246095, 19.5299, -8.28446e-2, 1.24692]

        self.best_weighted_errors = [0.0220313708486, 1.12879436221,
                                     0.0447659158681, 0.0412022938883]

        self.best_weighted_chisqr = 77.6040960351

        self.best_unweighted = [-0.10584111872702096, 19.240347049328989,
                                0.0092623066070940396, 1.501362314145845]

        self.best_unweighted_errors = [0.34246565477, 0.689820935208,
                                       0.0411243173041, 0.0693429375282]

        self.best_unweighted_chisqr = 497.102084956

        self.p0 = np.array([0.1, 20., 0.1, 0.1])
        self.bounds = [(-1, 1), (0, 30), (-5., 5.), (0.001, 2)]

        self.params = curvefitter.to_parameters(self.p0, bounds=self.bounds)

    def test_best_weighted(self):
        f = CurveFitter(gauss, (self.xvals, self.yvals, self.evals), self.params)
        res = f.fit()

        output = list(res.params.valuesdict().values())
        assert_almost_equal(output, self.best_weighted, 4)
        assert_almost_equal(res.chisqr, self.best_weighted_chisqr)

        uncertainties = [res.params['p%d' % i].stderr for i in range(4)]
        assert_almost_equal(uncertainties, self.best_weighted_errors, 3)

    def test_best_unweighted(self):
        f = CurveFitter(gauss,
                        (self.xvals, self.yvals),
                        self.params)
        res = f.fit()

        output = list(res.params.valuesdict().values())
        assert_almost_equal(output, self.best_unweighted, 5)
        assert_almost_equal(res.chisqr, self.best_unweighted_chisqr)

        uncertainties = [res.params['p%d' % i].stderr for i in range(4)]
        assert_almost_equal(uncertainties, self.best_unweighted_errors, 3)

    def test_pickleable(self):
        # residuals needs to be pickleable if one wants to use Pool
        f = CurveFitter(gauss, (self.xvals, self.yvals), self.params)
        import pickle
        pkl = pickle.dumps(f)
        pkl = pickle.dumps(f._resid)

    def test_parameter_names(self):
        # each instance of CurveFitter should be able to give a default set of
        # parameter names
        names = ['p%i' % i for i in range(10)]
        names2 = FitFunction.parameter_names(nparams=10)
        assert_(names == names2)

    def test_emcee_vs_lm(self):
        # test mcmc output vs lm
        f = CurveFitter(gauss,
                        (self.xvals, self.yvals, self.evals),
                        self.params)
        np.random.seed(123456)

        out = f.emcee(nwalkers=100, steps=500, burn=250, thin=20)
        within_sigma(self.best_weighted, out.params)
        # test if the sigmas are similar as well (within 20 %)
        errs = np.array([out.params[par].stderr for par in out.params])
        assert_allclose(errs, self.best_weighted_errors, rtol=0.2)

        # now try with resampling MC
        out = f._resampleMC(500, params=self.params, method='leastsq')
        within_sigma(self.best_weighted, out.params)
        # test if the sigmas are similar as well (within 20 %)
        errs = np.array([out.params[par].stderr for par in out.params])
        assert_allclose(errs, self.best_weighted_errors, rtol=0.2)

        # test mcmc output vs lm, some parameters not bounded
        self.params['p1'].max = np.inf
        f = CurveFitter(gauss,
                        (self.xvals, self.yvals, self.evals),
                        self.params)
        np.random.seed(123456)
        f.emcee(nwalkers=100, steps=300, burn=100, thin=5)
        within_sigma(self.best_weighted, out.params)

        # test mcmc output vs lm, some parameters not bounded
        self.params['p1'].min = -np.inf
        f = CurveFitter(gauss,
                        (self.xvals, self.yvals, self.evals),
                        self.params)
        f.emcee(nwalkers=100, steps=300, burn=100, thin=5)
        within_sigma(self.best_weighted, out.params)

    def test_lnpost(self):
        data = (self.xvals, self.yvals, self.evals)
        lnprob = _parallel_likelihood_calculator(self.params,
                                                 gauss,
                                                 data)

        def lnpost(pars, generative, y, e):
            resid = y - generative
            resid /= e
            resid *= resid
            resid += np.log(2 * np.pi * e**2)
            return -0.5 * np.sum(resid)

        lnprob2 = _parallel_likelihood_calculator(self.params,
                                                  gauss,
                                                  data,
                                                  lnpost=lnpost)

        assert_equal(lnprob2, lnprob)

        pars_copy = deepcopy(self.params)

        f = CurveFitter(gauss,
                        data,
                        self.params)
        res = f.emcee(steps=10, burn=0, thin=1, seed=1)

        g = CurveFitter(gauss,
                        data,
                        pars_copy,
                        lnpost=lnpost)
        res2 = g.emcee(steps=10, burn=0, thin=1, seed=1)
        assert_almost_equal(np.array(res.params), np.array(res2.params))


def within_sigma(desired, actual_params):
    # are the fitted params within sigma of where we know them to be?
    p0 = curvefitter.values(actual_params)
    sigmas = [actual_params[par].stderr for par in actual_params]
    for p, sigma, des in zip(p0, sigmas, desired):
        assert_allclose(p, des, atol=sigma)


if __name__ == '__main__':
    unittest.main()
