import unittest
import pyplatypus.analysis.reflect as reflect
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
            with values generated from Motofit
        '''
        theoretical = np.loadtxt('pyplatypus/analysis/test/smeared_theoretical.txt')
        qvals, rvals, dqvals = np.hsplit(theoretical, 3)
        calc = reflect.abeles(qvals.flatten(), self.coefs, **{'dqvals':dqvals.flatten()})
        
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
         
    def test_differential_evolutionfit(self):
        '''
            test differential evolution fitting process
        '''
        np.seterr(invalid='raise')
        theoretical = np.loadtxt('pyplatypus/analysis/test/c_PLP0011859_q.txt')

        qvals, rvals, evals = np.hsplit(theoretical, 3)
        
        coefs = np.zeros((16))
        coefs[0] = 2
        coefs[1] = 1.
        coefs[2] = 2.07
        coefs[4] = 6.36
        coefs[6] = 2e-6
        coefs[7] = 3
        coefs[8] = 300
        coefs[9] = 3.47
        coefs[11] = 4
        coefs[12] = 250
        coefs[13] = 2
        coefs[15] = 4
        
        fitted_parameters = np.array([3,5,6,7,8,9,10,11,12,13,14,15])
        
        a = reflect.ReflectivityFitObject(qvals, rvals, evals, coefs, fitted_parameters = fitted_parameters)
        a.fit()




if __name__ == '__main__':
    unittest.main()