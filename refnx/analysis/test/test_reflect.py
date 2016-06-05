import unittest
import refnx.analysis.reflect as reflect
try:
    import refnx.analysis._creflect as _creflect
except ImportError:
    HAVE_CREFLECT = False
else:
    HAVE_CREFLECT = True
import refnx.analysis._reflect as _reflect
import refnx.analysis.curvefitter as curvefitter
from refnx.analysis.curvefitter import CurveFitter
from refnx.analysis.reflect import ReflectivityFitFunction as RFF
from refnx.analysis.reflect import AnalyticalReflectivityFunction as ARF

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_allclose)
import os.path
import time


path = os.path.dirname(os.path.abspath(__file__))


class TestReflect(unittest.TestCase):

    def setUp(self):
        self.coefs = np.zeros(12)
        self.coefs[0] = 1.
        self.coefs[1] = 1.
        self.coefs[4] = 2.07
        self.coefs[7] = 3
        self.coefs[8] = 100
        self.coefs[9] = 3.47
        self.coefs[11] = 2

        self.layer_format = reflect.coefs_to_layer(self.coefs)

        theoretical = np.loadtxt(os.path.join(path, 'theoretical.txt'))
        qvals, rvals = np.hsplit(theoretical, 2)
        self.qvals = qvals.flatten()
        self.rvals = rvals.flatten()

        # e361 is an older dataset, but well characterised
        self.coefs361 = np.zeros(16)
        self.coefs361[0] = 2
        self.coefs361[1] = 1.
        self.coefs361[2] = 2.07
        self.coefs361[4] = 6.36
        self.coefs361[6] = 2e-5
        self.coefs361[7] = 3
        self.coefs361[8] = 10
        self.coefs361[9] = 3.47
        self.coefs361[11] = 4
        self.coefs361[12] = 200
        self.coefs361[13] = 1
        self.coefs361[15] = 3
        lowlim = np.zeros(16)
        lowlim[1] = 0.1
        lowlim[4] = 6.2
        hilim = 2 * self.coefs361

        bounds = list(zip(lowlim, hilim))
        e361 = np.loadtxt(os.path.join(path, 'e361r.txt'))
        self.qvals361, self.rvals361, self.evals361 = np.hsplit(e361, 3)
        np.seterr(invalid='raise')

        self.params361 = curvefitter.to_parameters(self.coefs361,
                                                   bounds=bounds,
                                                   varies=[False] * 16)
        fit = [1, 4, 6, 8, 12, 13]
        for p in fit:
            self.params361['p%d' % p].vary = True

    def test_abeles(self):
        # test reflectivity calculation with values generated from Motofit
        calc = reflect.reflectivity(self.qvals, self.coefs)

        assert_almost_equal(calc, self.rvals)

    def test_format_conversion(self):
        coefs = reflect.layer_to_coefs(self.layer_format)
        assert_equal(coefs, self.coefs)

    def test_c_abeles(self):
        if HAVE_CREFLECT:
            # test reflectivity calculation with values generated from Motofit
            calc = _creflect.abeles(self.qvals, self.layer_format)
            assert_almost_equal(calc, self.rvals)

            # test for non-contiguous Q values
            tempq = self.qvals[0::5]
            assert_(tempq.flags['C_CONTIGUOUS'] is False)
            calc = _creflect.abeles(tempq, self.layer_format)
            assert_almost_equal(calc, self.rvals[0::5])

    def test_py_abeles(self):
        # test reflectivity calculation with values generated from Motofit
        calc = _reflect.abeles(self.qvals, self.layer_format)
        assert_almost_equal(calc, self.rvals)

    def test_compare_c_py_abeles(self):
        # test python and c are equivalent
        # but not the same file
        if not HAVE_CREFLECT:
            return
        assert_(_reflect.__file__ != _creflect.__file__)

        calc1 = _reflect.abeles(self.qvals, self.layer_format)
        calc2 = _creflect.abeles(self.qvals, self.layer_format)
        assert_almost_equal(calc1, calc2)
        calc1 = _reflect.abeles(self.qvals, self.layer_format, scale=2.)
        calc2 = _creflect.abeles(self.qvals, self.layer_format, scale=2.)
        assert_almost_equal(calc1, calc2)
        calc1 = _reflect.abeles(self.qvals, self.layer_format, scale=0.5,
                                bkg=0.1)
        calc2 = _creflect.abeles(self.qvals, self.layer_format, scale=0.5,
                                 bkg=0.1)
        assert_almost_equal(calc1, calc2)

    def test_cabeles_parallelised(self):
        # I suppose this could fail if someone doesn't have a multicore computer
        if not HAVE_CREFLECT:
            return

        coefs = np.array([[0, 0, 0, 0],
                          [300, 3, 1e-3, 3],
                          [10, 3.47, 1e-3, 3],
                          [0, 6.36, 0, 3]])

        x = np.linspace(0.01, 0.2, 1000000)
        pstart = time.time()
        _creflect.abeles(x, coefs, parallel=True)
        pfinish = time.time()

        sstart = time.time()
        _creflect.abeles(x, coefs, parallel=False)
        sfinish = time.time()

        assert_(0.7 * (sfinish - sstart) > (pfinish - pstart))

    def test_compare_c_py_abeles0(self):
        # test two layer system
        if not HAVE_CREFLECT:
            return
        layer0 = np.array([[0, 2.07, 0.01, 3],
                           [0, 6.36, 0.1, 3]])
        calc1 = _reflect.abeles(self.qvals, layer0, scale=0.99, bkg=1e-8)
        calc2 = _creflect.abeles(self.qvals, layer0, scale=0.99, bkg=1e-8)
        assert_almost_equal(calc1, calc2)

    def test_compare_c_py_abeles2(self):
        # test two layer system
        if not HAVE_CREFLECT:
            return
        layer2 = np.array([[0, 2.07, 0.01, 3],
                           [10, 3.47, 0.01, 3],
                           [100, 1.0, 0.01, 4],
                           [0, 6.36, 0.1, 3]])
        calc1 = _reflect.abeles(self.qvals, layer2, scale=0.99, bkg=1e-8)
        calc2 = _creflect.abeles(self.qvals, layer2, scale=0.99, bkg=1e-8)
        assert_almost_equal(calc1, calc2)

    def test_c_abeles_reshape(self):
        # c reflectivity should be able to deal with multidimensional input
        if not HAVE_CREFLECT:
            return
        reshaped_q = np.reshape(self.qvals, (2, 250))
        reshaped_r = self.rvals.reshape(2, 250)
        calc = _creflect.abeles(reshaped_q, self.layer_format)
        assert_equal(reshaped_r.shape, calc.shape)
        assert_almost_equal(reshaped_r, calc, 15)

    def test_abeles_reshape(self):
        # reflectivity should be able to deal with multidimensional input
        reshaped_q = np.reshape(self.qvals, (2, 250))
        reshaped_r = self.rvals.reshape(2, 250)
        calc = _reflect.abeles(reshaped_q, self.layer_format)
        assert_equal(reshaped_r.shape, calc.shape)
        assert_almost_equal(reshaped_r, calc, 15)

    def test_reflectivity_model(self):
        # test reflectivity calculation with values generated from Motofit
        params = curvefitter.to_parameters(self.coefs)

        fitfunc = reflect.ReflectivityFitFunction(dq=0.)
        model = fitfunc.model(self.qvals, params)

        assert_almost_equal(model, self.rvals)

    def test_reflectivity_fit(self):
        # a smoke test to make sure the reflectivity fit proceeds
        fitfunc = reflect.ReflectivityFitFunction()
        transform = reflect.Transform('logY')
        yt, et = transform.transform(self.qvals361,
                                     self.rvals361,
                                     self.evals361)
        kws = {'transform': transform.transform}
        fitter2 = CurveFitter(fitfunc,
                              (self.qvals361, yt, et),
                              self.params361,
                              fcn_kws=kws,
                              kws={'seed': 2})
        fitter2.fit('differential_evolution')

    def test_reflectivity_emcee(self):
        transform = reflect.Transform('logY')
        yt, et = transform.transform(self.qvals361,
                                     self.rvals361,
                                     self.evals361)

        kws = {'transform': transform.transform}
        fitfunc = RFF(transform=transform.transform, dq=5.)

        fitter = CurveFitter(fitfunc,
                             (self.qvals361, yt, et),
                             self.params361,
                             fcn_kws=kws)
        res = fitter.fit()
        res_em = fitter.emcee(steps=10)
        # assert_allclose(values(res.params), values(res_em.params), rtol=1e-2)
        # for par in res.params:
        #     if res.params[par].vary:
        #         err = res.params[par].stderr
        #         em_err = res_em.params[par].stderr
        #         assert_allclose(err, em_err, rtol=0.1)

    def test_smearedabeles(self):
        # test smeared reflectivity calculation with values generated from
        # Motofit (quadrature precsion order = 13)
        theoretical = np.loadtxt(os.path.join(path, 'smeared_theoretical.txt'))
        qvals, rvals, dqvals = np.hsplit(theoretical, 3)
        '''
        the order of the quadrature precision used to create these smeared
        values in Motofit was 13.
        Do the same here
        '''
        calc = reflect.reflectivity(qvals.flatten(), self.coefs,
                                    **{'dqvals': dqvals.flatten(),
                                       'quad_order': 13})

        assert_almost_equal(rvals.flatten(), calc)

    def test_constant_smearing(self):
        # check that constant dq/q smearing is the same as point by point
        dqvals = 0.05 * self.qvals
        calc = reflect.reflectivity(self.qvals, self.coefs,
                                    **{'dqvals': dqvals,
                                       'quad_order': 'ultimate'})
        calc2 = reflect.reflectivity(self.qvals, self.coefs,
                                     **{'dqvals': 5.})

        assert_allclose(calc, calc2, rtol=0.011)

    def test_smearedabeles_reshape(self):
        # test smeared reflectivity calculation with values generated from
        # Motofit (quadrature precsion order = 13)
        theoretical = np.loadtxt(os.path.join(path, 'smeared_theoretical.txt'))
        qvals, rvals, dqvals = np.hsplit(theoretical, 3)
        '''
        the order of the quadrature precision used to create these smeared
        values in Motofit was 13.
        Do the same here
        '''
        reshaped_q = np.reshape(qvals, (2, 250))
        reshaped_r = np.reshape(rvals, (2, 250))
        reshaped_dq = np.reshape(dqvals, (2, 250))
        calc = reflect.reflectivity(reshaped_q, self.coefs,
                                    **{'dqvals': reshaped_dq,
                                       'quad_order': 13})

        assert_almost_equal(calc, reshaped_r, 15)

    def test_smeared_reflectivity_fitter(self):
        # test smeared reflectivity calculation with values generated from
        # Motofit (quadrature precsion order = 13)
        theoretical = np.loadtxt(os.path.join(path, 'smeared_theoretical.txt'))
        qvals, rvals, dqvals = np.hsplit(theoretical, 3)

        '''
        the order of the quadrature precision used to create these smeared
        values in Motofit was 13.
        Do the same here
        '''
        params = curvefitter.to_parameters(self.coefs)
        fitfunc = RFF(quad_order=13)
        fitter = CurveFitter(fitfunc,
                             (qvals, rvals),
                             params,
                             fcn_kws={'dqvals': dqvals})

        model = fitter.model(params)

        assert_almost_equal(model, rvals)

    def test_sld_profile(self):
        # test SLD profile with SLD profile from Motofit.
        np.seterr(invalid='raise')
        profile = np.loadtxt(os.path.join(path, 'sld_theoretical_R.txt'))
        z, rho = np.split(profile, 2)
        myrho = reflect.sld_profile(z.flatten(), self.coefs)
        assert_almost_equal(myrho, rho.flatten())

    def test_parameter_names(self):
        names = ['nlayers', 'scale', 'SLDfront', 'iSLDfront', 'SLDback',
                 'iSLDback', 'bkg', 'sigma_back']

        names += ['thick1', 'SLD1', 'iSLD1', 'sigma1']

        names2 = reflect.ReflectivityFitFunction.parameter_names(12)
        assert_(names == names2)


class AnalyticTestFunction(ARF):
    def to_slab(self, params):
        return [1, 1., 0, 0, 2.07, 0, 0, 3, 100, 3.47, 0, 2]


class TestAnalyticalProfile(unittest.TestCase):
    def setUp(self):
        self.arf = AnalyticTestFunction(dq=0)

        theoretical = np.loadtxt(os.path.join(path, 'theoretical.txt'))
        qvals, rvals = np.hsplit(theoretical, 2)
        self.qvals = qvals.flatten()
        self.rvals = rvals.flatten()

    def test_ARF_model(self):
        # simple smoke test to see if we can calculate reflectivity using
        # the analytical model setup
        rvals = self.arf.model(self.qvals, [2])
        assert_allclose(rvals, self.rvals)

if __name__ == '__main__':
    unittest.main()
