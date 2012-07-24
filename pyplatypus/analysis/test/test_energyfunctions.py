import unittest
import pyplatypus.analysis.reflect as reflect
import numpy as np
import pyplatypus.analysis.energyfunctions as energyfunctions
import pyplatypus.analysis.DEsolver as DEsolver
import numpy.testing as npt

class TestAnalysis(unittest.TestCase):

    def setUp(self):
        pass
                             
    def test_differential_evolutionfit(self):
        '''
            test differential evolution fitting process
        '''
        theoretical = np.loadtxt('pyplatypus/analysis/test/c_PLP0011859_q.txt')

        qvals, rvals, evals = np.hsplit(theoretical, 3)
        
        coefs = np.zeros((16))
        coefs[0] = 2
        coefs[1] = 1.
        coefs[2] = 2.07
        coefs[4] = 6.36
        coefs[7] = 3
        coefs[8] = 300
        coefs[9] = 3.47
        coefs[11] = 4
        coefs[12] = 250
        coefs[13] = 2
        coefs[15] = 4
        
        holdvector = np.zeros((16))
        holdvector[0] = 1
        holdvector[2] = 1
        holdvector[4] = 1
        holdvector[9] = 1
        
        a = energyfunctions.ReflectivityFitObject(qvals, rvals, evals, reflect.abeles, coefs, holdvector=holdvector)
        a.fit()
        

if __name__ == '__main__':
    unittest.main()