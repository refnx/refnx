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
        theoretical = np.loadtxt('theoretical.txt')
        qvals, rvals = np.hsplit(theoretical, 2)
        calc = reflect.abeles(qvals.flatten(), self.coefs)
        
        npt.assert_almost_equal(calc, rvals.flatten())
        
        #now do smeared calculation test
        theoretical = np.loadtxt('smeared_theoretical.txt')
        qvals, rvals, dqvals = np.hsplit(theoretical, 3)
        calc = reflect.abeles(qvals.flatten(), self.coefs, **{'dqvals':dqvals.flatten()})

        npt.assert_almost_equal(calc, rvals.flatten())
         

if __name__ == '__main__':
    unittest.main()