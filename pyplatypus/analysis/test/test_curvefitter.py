import unittest
import curvefitter
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

SEED = 1

def gauss(x, p, *args):
    return p[0] + p[1] * np.exp(-((x - p[2]) / p[3])**2)

class TestFitter(unittest.TestCase):

    def setUp(self):
        np.seterr(invalid='raise')
        self.xdata = np.linspace(-4, 4, 100)
        self.p0 = np.array([0., 1., 0.0, 1.])
        self.ydata = gauss(self.xdata, self.p0)
        self.f = curvefitter.CurveFitter(self.xdata, self.ydata, gauss, self.p0 + 0.2)
        self.bounds = [(-1, 1), (0, 2), (-1, 1.), (0.001, 2)]

    def test_fitting(self):
        '''the simplest test - a really simple gauss curve with perfect data'''
        res = self.f.fit()
        assert_almost_equal(res.p, self.p0)
        assert_almost_equal(res.cost, 0)

    def test_covp_size(self):
        '''cov_p should have p0.size rows and p0.size columns'''
        res = self.f.fit()
        assert_equal(np.size(res.cov_p, 0), self.f.p0.size)

    def test_model_returns_function(self):
        ydata = gauss(self.xdata, self.p0)
        model = self.f.model(self.p0)
        assert_almost_equal(ydata, model)

    def test_residuals(self):
        resid = self.f.residuals(self.p0)
        assert_almost_equal(resid, 0)
        resid = self.f.residuals(np.array([0., 0., 0., 1.]))
        assert_almost_equal(resid, self.ydata)

    def test_cost(self):
        resid = self.f.residuals(np.array([0., 0., 0., 1.]))
        cost = self.f.cost(np.array([0., 0., 0., 1.]))
        assert_almost_equal(cost, np.sum(self.ydata**2))

    def test_custom_scipy_minimize(self):
        '''test that a custom method can be used with scipy.optimize.minimize'''
        minimizer_kwds = {'bounds':self.bounds}
        res = self.f.fit(method=curvefitter.de_wrapper,
                         minimizer_kwds=minimizer_kwds)
        assert_almost_equal(res.p, self.p0)

    def test_minimize(self):
        '''test that scipy.optimize.minimize methods can be used'''
        minimizer_kwds = {'bounds':self.bounds}
        res = self.f.fit(method='L-BFGS-B', minimizer_kwds=minimizer_kwds)
        assert_almost_equal(res.p, self.p0, 3)

    def test_holding_parameter(self):
        '''
        holding parameters means that those parameters shouldn't change
        during a fit
        '''
        p = self.f.fit(pheld=[0]).p
        assert_almost_equal(p[0], self.f.p0[0])

if __name__ == '__main__':
    unittest.main()