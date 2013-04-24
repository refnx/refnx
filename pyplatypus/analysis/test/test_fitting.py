import unittest
import pyplatypus.analysis.reflect as reflect
import pyplatypus.analysis.Model as model
import numpy as np
import numpy.testing as npt

SEED = 1

class TestDifferentialEvolution(unittest.TestCase):

    def setUp(self):
        np.seterr(invalid='raise')
        theoretical = np.loadtxt('pyplatypus/analysis/test/c_PLP0011859_q.txt')

        self.qvals, self.rvals, self.evals, dummy = np.hsplit(theoretical, 4)
        self.coefs = np.zeros((16))
        self.coefs[0] = 2
        self.coefs[1] = 1.
        self.coefs[2] = 2.07
        self.coefs[4] = 6.36
        self.coefs[6] = 2e-6
        self.coefs[7] = 3
        self.coefs[8] = 30
        self.coefs[9] = 3.47
        self.coefs[11] = 4
        self.coefs[12] = 250
        self.coefs[13] = 2
        self.coefs[15] = 4
        self.fitted_parameters = np.array([6, 7,8,11,12,13,15])

        self.a = reflect.ReflectivityFitObject(self.qvals,
                                                     self.rvals,
                                                      self.evals,
                                                       self.coefs,
                                                        fitted_parameters = self.fitted_parameters, seed = SEED)
        
    def test_differential_evolutionfit(self):
        '''
            test differential evolution fitting process
        '''
        np.seterr(invalid='raise')
        pars, dummy, chi2 = self.a.fit()
        
#         modeltosave = model.Model(pars, limits = a.limits, fitted_parameters = fitted_parameters)
#         with open('pyplatypus/analysis/test/fittedcoefs_11859.txt', 'w') as f:
#             modeltosave.save(f)
            
        with open('pyplatypus/analysis/test/fittedcoefs_11859.txt', 'Ur') as f:
            savedmodel = model.Model(None, file = f)

    #    saved_pars = np.load('pyplatypus/analysis/test/fittedcoefs_11859.npy')
        npt.assert_almost_equal(pars, savedmodel.parameters)

    def test_LMfit(self):
        '''
            test Levenberg-Marquardt fitting process
        '''
        
        #need to redo the setup because the original fit will have changed the fit parameters
        self.setUp()
        pars, dummy, chi2 = self.a.fit(method = 'LM')

#         modeltosave = model.Model(pars, limits = a.limits, fitted_parameters = fitted_parameters)
#         with open('pyplatypus/analysis/test/fittedcoefs_11859.txt', 'w') as f:
#             modeltosave.save(f)
            
        with open('pyplatypus/analysis/test/fittedcoefs_11859.txt', 'Ur') as f:
            savedmodel = model.Model(None, file = f)
        
#        npt.assert_almost_equal(pars, savedmodel.parameters)
        npt.assert_almost_equal(chi2, 2974.5455826362172)



if __name__ == '__main__':
    unittest.main()