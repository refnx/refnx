import unittest
import pyplatypus.analysis.curvefitter as curvefitter
from pyplatypus.analysis.curvefitter import CurveFitter
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_

SEED = 1

def gauss(x, p0, *args):
    p = p0.valuesdict().values()
    return p[0] + p[1] * np.exp(-((x - p[2]) / p[3])**2)

class TestFitter(unittest.TestCase):

    def setUp(self):
        np.seterr(invalid='raise')
        self.xdata = np.linspace(-4, 4, 100)
        self.p0 = np.array([0., 1., 0.0, 1.])
        self.bounds = [(-1, 1), (0, 2), (-1, 1.), (0.001, 2)]

        self.params = curvefitter.params(self.p0 + 0.2, bounds=self.bounds)
        self.final_params = curvefitter.params(self.p0, bounds=self.bounds)

        self.ydata = gauss(self.xdata, self.final_params)
        self.f = CurveFitter(self.params, self.xdata, self.ydata, gauss)

    def pvals(self, params):
        return np.asfarray(params.valuesdict().values())

    def test_fitting(self):
        #the simplest test - a really simple gauss curve with perfect data
        res = self.f.fit()
        assert_(res, 'True')
        assert_almost_equal(self.pvals(self.params), self.p0)
        assert_almost_equal(self.f.chisqr, 0)

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
        #test that a custom method can be used with scipy.optimize.minimize
        self.f.fit()
        assert_almost_equal(self.pvals(self.params), self.p0)
        
    def test_resid_length(self):
        # the residuals length should be equal to the data length
        resid = self.f.residuals(self.params)
        assert_equal(resid.size, self.f.ydata)

    def test_scalar_minimize(self):
        assert_equal(self.pvals(self.params), self.p0 + 0.2)
        self.f.fit(method='differential_evolution')
        assert_almost_equal(self.pvals(self.params), self.p0, 3)

    def test_holding_parameter(self):
        #holding parameters means that those parameters shouldn't change
        #during a fit
        self.params['p0'].vary = False
        self.f.fit()
        assert_almost_equal(self.p0[0] + 0.2, self.params['p0'].value)

if __name__ == '__main__':
    unittest.main()