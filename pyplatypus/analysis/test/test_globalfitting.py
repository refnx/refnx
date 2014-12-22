import unittest
import pyplatypus.analysis.reflect as reflect
import pyplatypus.analysis.curvefitter as curvefitter
from pyplatypus.analysis.curvefitter import GlobalFitter, CurveFitter
import numpy as np
from lmfit import fit_report
import os.path
from numpy.testing import assert_, assert_equal, assert_almost_equal
import time

SEED = 1

path = os.path.dirname(os.path.abspath(__file__))

def reflect_fitfunc(q, params, *args):
    coefs = np.asfarray(params.valuesdict().values())
    return np.log10(reflect.abeles(q, coefs))

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

        self.best_fit = np.array([2, 8.9692702e-01, 2.07, 0., 6.36, 0.,
                                  3.3588842e-07, 2.8938204, 38.128943,
                                  3.47, 0., 3.0, 2.5909985e+02, 2.5406819e+00,
                                  0., 3.])

        lowlim = np.zeros(16)
        hilim = 2 * coefs

        self.bounds = zip(lowlim, hilim)
        self.params = curvefitter.params(coefs, bounds=self.bounds,
                                         varies=[False] * 16)

        fname = os.path.join(path, 'c_PLP0011859_q.txt')
        theoretical = np.loadtxt(fname)
        qvals, rvals, evals, dummy = np.hsplit(theoretical, 4)
        rvals = np.log10(rvals)
        self.f = curvefitter.CurveFitter(self.params, qvals.flatten(),
                                         rvals.flatten(), reflect_fitfunc)

    def test_residuals_length(self):
        # the residuals should be the same length as the data
        a = GlobalFitter([self.f])
        residuals = a.residuals(a.params)
        assert_equal(residuals.size, a.datasets[0].ydata.size)

    def test_globalfitting(self):
        # can the global fitting run?
        fit = [1, 6, 7, 8, 12, 13]
        for p in fit:
            self.params['p%d' % p].vary = True

        a = GlobalFitter([self.f], minimizer_kwds={'options':{'seed':1}})
        a.fit(method='differential_evolution')

        values = self.params.valuesdict().values()
        assert_almost_equal(values, self.best_fit, 4)

    def test_globfit_modelvals_same_as_indidivual(self):
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
        self.bounds = zip(lowlim, hilim)
        params = curvefitter.params(coefs, bounds=self.bounds,
                                         varies=[False] * 20)

        fit = np.array([6, 7, 8, 11, 12, 13, 15, 16, 17, 19])
        for p in fit:
            params['p%d' % p].vary = True

        self.f.params = params
        a = GlobalFitter([self.f], constraints=['d0p16:d0p12', 'd0p17:d0p13',
                         'd0p19:d0p15'])

        a.fit(method='differential_evolution')

        values = params.valuesdict().values()

        assert_equal(values[12], values[16])
        assert_equal(values[13], values[17])
        assert_equal(values[15], values[19])

    def test_multipledataset_corefinement(self):
        # test corefinement of three datasets
        e361 = np.loadtxt(os.path.join(path, 'e361r.txt'))
        e365 = np.loadtxt(os.path.join(path, 'e365r.txt'))
        e366 = np.loadtxt(os.path.join(path, 'e366r.txt'))

        coefs361 = np.zeros((16))
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
        bounds = zip(lowlim, hilim)

        params361 = curvefitter.params(coefs361, bounds=bounds,
                                       varies=[False] * 16)
        params365 = curvefitter.params(coefs365, bounds=bounds,
                                       varies=[False] * 16)
        params366 = curvefitter.params(coefs366, bounds=bounds,
                                       varies=[False] * 16)

        fit = [1, 6, 8, 12, 13]
        for p in fit:
            params361['p%d' % p].vary = True
            params365['p%d' % p].vary = True
            params366['p%d' % p].vary = True

        a = CurveFitter(params361, qvals361.flatten(),
                        np.log10(rvals361.flatten()),
                        reflect_fitfunc)
        b = CurveFitter(params365, qvals365.flatten(),
                        np.log10(rvals365.flatten()),
                        reflect_fitfunc)
        c = CurveFitter(params366, qvals366.flatten(),
                        np.log10(rvals366.flatten()),
                        reflect_fitfunc)

        g = GlobalFitter([a, b, c], constraints=['d1p8:d0p8', 'd2p8:d0p8',
                         'd1p12:d0p12', 'd2p12:d0p12'],
                         minimizer_kwds={'options':{'seed':1}})
        
        g.fit('differential_evolution')
        #print fit_report(g)
        assert_almost_equal(g.chisqr, 0.774590447535)


if __name__ == '__main__':
    unittest.main()
