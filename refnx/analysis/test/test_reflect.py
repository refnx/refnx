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

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_,\
                           assert_allclose)
import os.path
import time

path = os.path.dirname(os.path.abspath(__file__))


class TestReflect(unittest.TestCase):

    def setUp(self):
        self.coefs = np.zeros((12))
        self.coefs[0] = 1.
        self.coefs[1] = 1.
        self.coefs[4] = 2.07
        self.coefs[7] = 3
        self.coefs[8] = 100
        self.coefs[9] = 3.47
        self.coefs[11] = 2

        self.layer_format = reflect.convert_coefs_to_layer_format(self.coefs)

        theoretical = np.loadtxt(os.path.join(path, 'theoretical.txt'))
        qvals, rvals = np.hsplit(theoretical, 2)
        self.qvals = qvals.flatten()
        self.rvals = rvals.flatten()

    def test_abeles(self):
        #    test reflectivity calculation with values generated from Motofit
        calc = reflect.abeles(self.qvals, self.coefs)

        assert_almost_equal(calc, self.rvals)

    def test_format_conversion(self):
        coefs = reflect.convert_layer_format_to_coefs(self.layer_format)
        assert_equal(coefs, self.coefs)

    def test_c_abeles(self):
        if HAVE_CREFLECT:
            #    test reflectivity calculation with values generated from Motofit
            calc = _creflect.abeles(self.qvals, self.layer_format)
            assert_almost_equal(calc, self.rvals)

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

    def test_compare_c_py_abeles0(self):
        #test two layer system
        if not HAVE_CREFLECT:
            return
        layer0 = np.array([[0, 2.07, 0.01, 3],
                           [0, 6.36, 0.1, 3]])
        calc1 = _reflect.abeles(self.qvals, layer0, scale=0.99, bkg=1e-8)
        calc2 = _creflect.abeles(self.qvals, layer0, scale=0.99, bkg=1e-8)
        assert_almost_equal(calc1, calc2)

    def test_compare_c_py_abeles2(self):
        #test two layer system
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
        # c abeles should be able to deal with multidimensional input
        if not HAVE_CREFLECT:
            return
        reshaped_q = np.reshape(self.qvals, (2, 250))
        reshaped_r = self.rvals.reshape(2, 250)
        calc = _creflect.abeles(reshaped_q, self.layer_format)
        assert_equal(reshaped_r.shape, calc.shape)
        assert_almost_equal(reshaped_r, calc, 15)

    def test_abeles_reshape(self):
        # abeles should be able to deal with multidimensional input
        reshaped_q = np.reshape(self.qvals, (2, 250))
        reshaped_r = self.rvals.reshape(2, 250)
        calc = _reflect.abeles(reshaped_q, self.layer_format)
        assert_equal(reshaped_r.shape, calc.shape)
        assert_almost_equal(reshaped_r, calc, 15)

    def test_reflectivity_model(self):
        # test reflectivity calculation with values generated from Motofit
        params = curvefitter.to_Parameters(self.coefs)

        fitter = reflect.ReflectivityFitter(self.qvals, self.rvals, params)
        model = fitter.model(params)

        assert_almost_equal(model, self.rvals)

    def test_reflectivity_fit(self):
        params = curvefitter.to_Parameters(self.coefs)

        fitter = reflect.ReflectivityFitter(self.qvals, self.rvals, params)
        fitter.fit()

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
        calc = reflect.abeles(qvals.flatten(), self.coefs,
                              **{'dqvals': dqvals.flatten(), 'quad_order': 13})

        assert_almost_equal(rvals.flatten(), calc)

    def test_constant_smearing(self):
        #check that constant dq/q smearing is the same as point by point
        dqvals = 0.05 * self.qvals
        calc = reflect.abeles(self.qvals, self.coefs,
                              **{'dqvals': dqvals, 'quad_order': 'ultimate'})
        calc2 = reflect.abeles(self.qvals, self.coefs,
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
        calc = reflect.abeles(reshaped_q, self.coefs,
                              **{'dqvals': reshaped_dq, 'quad_order': 13})

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
        params = curvefitter.to_Parameters(self.coefs)

        fitter = reflect.ReflectivityFitter(qvals,
                                        rvals, params,
                                        fcn_kws={'dqvals': dqvals.flatten(),
                                       'quad_order': 13})
        model = fitter.model(params)

        assert_almost_equal(model, rvals)

    def test_sld_profile(self):
        # test SLD profile with SLD profile from Motofit.
        np.seterr(invalid='raise')
        profile = np.loadtxt(os.path.join(path, 'sld_theoretical_R.txt'))
        z, rho = np.split(profile, 2)
        myrho = reflect.sld_profile(self.coefs, z.flatten())
        assert_almost_equal(myrho, rho.flatten())


if __name__ == '__main__':
    unittest.main()
