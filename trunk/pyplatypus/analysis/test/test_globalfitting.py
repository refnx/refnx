import unittest
import pyplatypus.analysis.reflect as reflect
import pyplatypus.analysis.globalfitting as gfit
import numpy as np
import numpy.testing as npt

class TestGlobalFitting(unittest.TestCase):

    def setUp(self):
        pass
                 
    def test_globalfitting(self):
        '''
            test differential evolution fitting process
        '''
        np.seterr(invalid='raise')
        theoretical = np.loadtxt('pyplatypus/analysis/test/c_PLP0011859_q.txt')

        qvals, rvals, evals, dummy = np.hsplit(theoretical, 4)

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
        linkageArray = np.arange(16)
        
        gfo = gfit.GlobalFitObject(tuple([a]), linkageArray)
        gfo.fit()
        
    def test_globfit_modelvals_same_as_indidivual(self):
        '''
            make sure that the global fit would return the same model values as the individual fitobject
        '''
        np.seterr(invalid='raise')
        theoretical = np.loadtxt('pyplatypus/analysis/test/c_PLP0011859_q.txt')

        qvals, rvals, evals, dummy = np.hsplit(theoretical, 4)

        coefs = np.zeros((16))
        coefs[0] = 2
        coefs[1] = 1.
        coefs[2] = 2.07
        coefs[4] = 6.36
        coefs[6] = 2e-6
        coefs[7] = 3
        coefs[8] = 30
        coefs[9] = 3.47
        coefs[11] = 4
        coefs[12] = 250
        coefs[13] = 2
        coefs[15] = 4
        
        fitted_parameters = np.array([3,5,6,7,8,9,10,11,12,13,14,15])
        
        a = reflect.ReflectivityFitObject(qvals, rvals, evals, coefs, fitted_parameters = fitted_parameters)
        linkageArray = np.arange(16)
        
        gfo = gfit.GlobalFitObject(tuple([a]), linkageArray)
        gfomodel = gfo.model(coefs)
        
        normalmodel = a.model(coefs)
        npt.assert_almost_equal(gfomodel, normalmodel)        

        
    def test_globfit_modelvals_degenerate_layers(self):
        '''
            try fitting dataset with a deposited layer split into two degenerate layers
        '''
        np.seterr(invalid='raise')
        theoretical = np.loadtxt('pyplatypus/analysis/test/c_PLP0011859_q.txt')

        qvals, rvals, evals, dummy = np.hsplit(theoretical, 4)

        coefs = np.zeros((20))
        coefs[0] = 3
        coefs[1] = 1.
        coefs[2] = 2.07
        coefs[4] = 6.36
        coefs[6] = 2e-6
        coefs[7] = 3
        coefs[8] = 30
        coefs[9] = 3.47
        coefs[11] = 4
        coefs[12] = 125
        coefs[13] = 2
        coefs[15] = 4
        coefs[16] = 125
        coefs[17] = 2
        coefs[19] = 4

        
        fitted_parameters = np.array([6,7,8,11,12,13,15,16,17,19])
        
        a = reflect.ReflectivityFitObject(qvals, rvals, evals, coefs, fitted_parameters = fitted_parameters)
        linkageArray = np.arange(20)
        linkageArray[16] = 12
        linkageArray[17] = 16
        linkageArray[18] = 17
        linkageArray[19] = 18
    
        gfo = gfit.GlobalFitObject(tuple([a]), linkageArray)
        pars, dummy, chi2 = gfo.fit() 
        npt.assert_almost_equal(pars[12], pars[16])

    def test_linkageArray(self):
        '''
            test incorrect linkageArrays
        '''
        np.seterr(invalid='raise')
        theoretical = np.loadtxt('pyplatypus/analysis/test/c_PLP0011859_q.txt')

        qvals, rvals, evals, dummy = np.hsplit(theoretical, 4)

        coefs = np.zeros((20))
        coefs[0] = 3
        coefs[1] = 1.
        coefs[2] = 2.07
        coefs[4] = 6.36
        coefs[6] = 2e-6
        coefs[7] = 3
        coefs[8] = 30
        coefs[9] = 3.47
        coefs[11] = 4
        coefs[12] = 125
        coefs[13] = 2
        coefs[15] = 4
        coefs[16] = 125
        coefs[17] = 2
        coefs[19] = 4
        
        fitted_parameters = np.array([6,7,8,11,12,13,15,16,17,19])
        
        a = reflect.ReflectivityFitObject(qvals, rvals, evals, coefs, fitted_parameters = fitted_parameters)
        linkageArray = np.arange(20)
        linkageArray[16] = 12
        linkageArray[17] = 15
        linkageArray[18] = 17
        linkageArray[19] = 18

        npt.assert_raises(Exception, gfit.GlobalFitObject, tuple([a]), linkageArray)
        linkageArray[17] = 16
        linkageArray[19] = -1
        npt.assert_raises(Exception, gfit.GlobalFitObject, tuple([a]), linkageArray)
        linkageArray[5] = -1
        npt.assert_raises(Exception, gfit.GlobalFitObject, tuple([a]), linkageArray)
        

if __name__ == '__main__':
    unittest.main()