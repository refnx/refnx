import unittest
import pyplatypus.analysis.reflect as reflect
import numpy as np
import pyplatypus.analysis.energyfunctions as energyfunctions
import pyplatypus.analysis.DEsolver as DEsolver
import numpy.testing as npt

class TestAnalysis(unittest.TestCase):

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
        
    def test_differential_evolutionfit(self):
        '''
            test differential evolution fitting process
        '''
        theoretical = np.loadtxt('c_PLP0011859_q.txt')

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
        
        a = energyfunctions.ReflectivityFitObject(qvals, rvals, evals, reflect.abeles, coefs)
        limits = np.zeros((2,16))
        limits[1:, ] = 2*coefs
        limits[:,0] = 2
        de = DEsolver.DEsolver(limits, energyfunctions.energy_for_fitting, a)
        print de.solve()
        

if __name__ == '__main__':
    unittest.main()