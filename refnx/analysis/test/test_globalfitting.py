import unittest
import refnx.analysis.reflect as reflect
import refnx.analysis.curvefitter as curvefitter
from refnx.analysis.curvefitter import GlobalFitter, CurveFitter
import numpy as np
#from lmfit import fit_report, Parameters
import os.path
from numpy.testing import assert_, assert_equal, assert_almost_equal

SEED = 1

CURDIR = os.path.dirname(os.path.abspath(__file__))


def reflect_fitfunc(q, params, *args):
    coefs = np.asfarray(list(params.valuesdict().values()))
    return np.log10(reflect.reflectivity(q, coefs, parallel=True))

    
class Test_reflect(unittest.TestCase):
    def setUp(self):
        self.coefs = np.zeros((12))
        self.coefs[0] = 1.
        self.coefs[1] = 1.
        self.coefs[4] = 2.07
        self.coefs[7] = 3
        self.coefs[8] = 100
        self.coefs[9] = 3.47
        self.coefs[11] = 2
        
        self.layer_format = reflect.coefs_to_layer(self.coefs)

        theoretical = np.loadtxt(os.path.join(CURDIR, 'theoretical.txt'))
        qvals, rvals = np.hsplit(theoretical, 2)
        self.qvals = qvals.flatten()
        self.rvals = rvals.flatten()

    def test_abeles(self):
        # test reflectivity calculation with values generated from Motofit
        p = curvefitter.to_parameters(self.coefs)
        calc = reflect_fitfunc(self.qvals, p)
        calc = np.power(10, calc)
        assert_almost_equal(calc, self.rvals)


class TestGlobalFitting(unittest.TestCase):

    def setUp(self):
        coefs = np.zeros((16))
        coefs[0] = 2
        coefs[1] = 1.
        coefs[2] = 2.07
        coefs[4] = 6.36
        coefs[6] = 2.e-07
        coefs[7] = 3
        coefs[8] = 40
        coefs[9] = 3.47
        coefs[11] = 3
        coefs[12] = 200.
        coefs[13] = 2
        coefs[15] = 3
        self.coefs = coefs

        self.best_fit = np.array([2, 8.9692e-01, 2.07, 0., 6.36, 0.,
                                  3.3588842e-07, 2.8938204, 38.129,
                                  3.47, 0., 3.0, 2.5910e+02, 2.5406,
                                  0., 3.])

        lowlim = np.zeros(16)
        hilim = 2 * coefs
        self.params = curvefitter.to_parameters(coefs, bounds=zip(lowlim, hilim),
                                         varies=[False] * 16)

        fname = os.path.join(CURDIR, 'c_PLP0011859_q.txt')

        theoretical = np.loadtxt(fname)
        qvals, rvals, evals, dummy = np.hsplit(theoretical, 4)
        rvals = np.log10(rvals)
        self.f = curvefitter.CurveFitter(reflect_fitfunc,
                                         (qvals.flatten(), rvals.flatten()),
                                         self.params)
        
    def test_residuals_length(self):
        # the residuals should be the same length as the data
        a = GlobalFitter([self.f])
        residuals = a.residuals(a.params)
        assert_equal(residuals.size, a.fitters[0].ydata.size)

    def test_globalfitting(self):
        # can the global fitting run?
        fit = [1, 6, 7, 8, 12, 13]
        for p in fit:
            self.params['p%d' % p].vary = True

        a = GlobalFitter([self.f], kws={'seed': 1})
        a.fit(method='differential_evolution')

        values = list(self.params.valuesdict().values())
        assert_almost_equal(values, self.best_fit, 3)

    def test_globfit_modelvals_same_as_individual(self):
        # make sure that the global fit would return the same model values as
        # the individual fitobject
        values = self.f.model(self.params)

        a = GlobalFitter([self.f])
        values2 = a.model(a.params)

        assert_almost_equal(values2, values)

    def test_globfit_modelvals_degenerate_layers(self):
        # try fitting dataset with a deposited layer split into two degenerate layers
        coefs = np.zeros((20))
        coefs[0] = 3
        coefs[1] = 1.
        coefs[2] = 2.07
        coefs[4] = 6.36
        coefs[6] = 2e-6
        coefs[7] = 3
        coefs[8] = 30
        coefs[9] = 3.47
        coefs[11] = 4
        coefs[12] = 125
        coefs[13] = 2
        coefs[15] = 4
        coefs[16] = 125
        coefs[17] = 2
        coefs[19] = 4

        lowlim = np.zeros(20)
        hilim = 2 * coefs
        bounds = zip(lowlim, hilim)
        params = curvefitter.to_parameters(coefs, bounds=bounds,
                                         varies=[False] * 20)

        fit = np.array([6, 7, 8, 11, 12, 13, 15, 16, 17, 19])
        for p in fit:
            params['p%d' % p].vary = True

        self.f.params = params
        a = GlobalFitter([self.f], constraints=['d0:p16=d0:p12',
                                                'd0:p17=d0:p13',
                                                'd0:p19=d0:p15'])

        res = a.fit(method='differential_evolution')

        values = list(res.params.valuesdict().values())

        assert_equal(values[12], values[16])
        assert_equal(values[13], values[17])
        assert_equal(values[15], values[19])

    def test_multipledataset_corefinement(self):
        # test corefinement of three datasets
        e361 = np.loadtxt(os.path.join(CURDIR, 'e361r.txt'))
        e365 = np.loadtxt(os.path.join(CURDIR, 'e365r.txt'))
        e366 = np.loadtxt(os.path.join(CURDIR, 'e366r.txt'))

        coefs361 = np.zeros(16)
        coefs361[0] = 2
        coefs361[1] = 1.
        coefs361[2] = 2.07
        coefs361[4] = 6.36
        coefs361[6] = 2e-5
        coefs361[7] = 3
        coefs361[8] = 10
        coefs361[9] = 3.47
        coefs361[11] = 4
        coefs361[12] = 200
        coefs361[13] = 1
        coefs361[15] = 3

        coefs365 = np.copy(coefs361)
        coefs366 = np.copy(coefs361)
        coefs365[4] = 3.47
        coefs366[4] = -0.56

        qvals361, rvals361, evals361 = np.hsplit(e361, 3)
        qvals365, rvals365, evals365 = np.hsplit(e365, 3)
        qvals366, rvals366, evals366 = np.hsplit(e366, 3)

        lowlim = np.zeros(16)
        lowlim[4] = -0.8
        hilim = 2 * coefs361
        
        bounds = list(zip(lowlim, hilim))
        params361 = curvefitter.to_parameters(coefs361, bounds=bounds,
                                       varies=[False] * 16)
        params365 = curvefitter.to_parameters(coefs365, bounds=bounds,
                                       varies=[False] * 16)
        params366 = curvefitter.to_parameters(coefs366, bounds=bounds,
                                       varies=[False] * 16)
        assert_(len(params361), 16)
        assert_(len(params365), 16)
        assert_(len(params366), 16)

        fit = [1, 6, 8, 12, 13]
        for p in fit:
            params361['p%d' % p].vary = True
            params365['p%d' % p].vary = True
            params366['p%d' % p].vary = True

        a = CurveFitter(reflect_fitfunc,
                        (qvals361.flatten(), np.log10(rvals361.flatten())),
                        params361)
        b = CurveFitter(reflect_fitfunc,
                        (qvals365.flatten(), np.log10(rvals365.flatten())),
                        params365)
        c = CurveFitter(reflect_fitfunc,
                        (qvals366.flatten(), np.log10(rvals366.flatten())),
                        params366)

        g = GlobalFitter([a, b, c], constraints=['d1:p8=d0:p8',
                                                 'd2:p8=d0:p8',
                                                 'd1:p12=d0:p12',
                                                 'd2:p12 = d0:p12'],
                         kws={'seed':1})


        indiv_chisqr = (a.residuals(a.params) ** 2
                        + b.residuals(b.params) ** 2
                        + c.residuals(c.params) ** 2)
        global_chisqr = g.residuals(g.params) ** 2
        assert_almost_equal(indiv_chisqr.sum(), global_chisqr.sum())
        # import time
        res = g.fit('differential_evolution')
        # start = time.time()
        # g.emcee(params=res.params, nwalkers=300, steps=500, workers=1)
        # finish = time.time()
        # print(finish - start)
        assert_almost_equal(res.chisqr, 0.774590447535, 4)

        # updating of constraints should happen during the fit
        assert_almost_equal(a.params['p12'].value, res.params['p12_d0'].value)
        assert_almost_equal(b.params['p12'].value, a.params['p12'].value)
        assert_almost_equal(c.params['p12'].value, a.params['p12'].value)

        g.params['p8_d0'].value=10.123456
        # shouldn't need to call update constraints within the gfitter, that
        # happens when you retrieve a specific value
        assert_almost_equal(g.params['p8_d1'].value, g.params['p8_d0'].value)
        # However, you have to call model or residuals to redistribute the
        # parameters to the original fitters
        g.model()
        assert_almost_equal(a.params['p8'].value, 10.123456)
        assert_almost_equal(b.params['p8'].value, 10.123456)


if __name__ == '__main__':
    unittest.main()
