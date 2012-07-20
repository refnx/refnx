import unittest
import pyplatypus.analysis.reflect as reflect
import numpy as np
import numpy.testing as npt

class TestReflectivityCalculation(unittest.TestCase):

    def setUp(self):
        self.coefs = np.zeros((12))
        self.coefs[0] = 1.
        self.coefs[1] = 1.
        self.coefs[4] = 2.07
        self.coefs[7] = 3
        self.coefs[8] = 100
        self.coefs[9] = 3.47
        self.coefs[11] = 2
        
        theoretical = np.loadtxt('theoretical.txt')
        qvals, rvals = np.hsplit(theoretical, 2)
        self.qvals = qvals.flatten()
        self.rvals = rvals.flatten()
     
    def test_abeles(self):
        calc = reflect.abeles(self.coefs, self.qvals)
        npt.assert_almost_equal(calc, self.rvals)

if __name__ == '__main__':
    unittest.main()