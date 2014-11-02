import unittest
import pyplatypus.analysis.reflect as reflect
import pyplatypus.analysis._creflect as _creflect

import numpy as np
import numpy.testing as npt


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

    def test_abeles(self):
        '''
            test reflectivity calculation
            with values generated from Motofit

        '''
        theoretical = np.loadtxt('pyplatypus/analysis/test/theoretical.txt')
        qvals, rvals = np.hsplit(theoretical, 2)
        calc = reflect.abeles(qvals.flatten(), self.coefs)

        npt.assert_almost_equal(calc, rvals.flatten())

    def test_smearedabeles(self):
        '''
            test smeared reflectivity calculation
            with values generated from Motofit (quadrature precsion order = 13)
        '''
        theoretical = np.loadtxt(
            'pyplatypus/analysis/test/smeared_theoretical.txt')
        qvals, rvals, dqvals = np.hsplit(theoretical, 3)
        '''
        the order of the quadrature precision used to create these smeared
        values in Motofit was 13.
        Do the same here
        '''
        calc = reflect.abeles(qvals.flatten(), self.coefs,
                              **{'dqvals': dqvals.flatten(), 'quad_order': 13})

        npt.assert_almost_equal(calc, rvals.flatten())

    def test_sld_profile(self):
        '''
            test SLD profile with SLD profile from Motofit.
        '''
        np.seterr(invalid='raise')
        profile = np.loadtxt('pyplatypus/analysis/test/sld_theoretical_R.txt')
        z, rho = np.split(profile, 2)
        myrho = reflect.sld_profile(self.coefs, z.flatten())
        npt.assert_almost_equal(myrho, rho.flatten())


if __name__ == '__main__':
    unittest.main()
