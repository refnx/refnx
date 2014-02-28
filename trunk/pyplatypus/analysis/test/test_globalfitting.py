import unittest
import pyplatypus.analysis.reflect as reflect
import pyplatypus.analysis.model as model
import pyplatypus.analysis.globalfitting as gfit
import numpy as np
import numpy.testing as npt
import warnings

SEED = 1


class TestGlobalFitting(unittest.TestCase):

    def setUp(self):
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
        self.coefs = coefs
        pass

    def test_globalfitting(self):
        '''
            test differential evolution fitting process on globalfit object
        '''
        theoretical = np.loadtxt('pyplatypus/analysis/test/c_PLP0011859_q.txt')

        qvals, rvals, evals, dummy = np.hsplit(theoretical, 4)

        fitted_parameters = np.array([6, 7, 8, 9, 11, 12, 13, 15])

        a = reflect.ReflectivityFitObject(
            qvals, rvals, evals, self.coefs, fitted_parameters=fitted_parameters, seed=SEED)
        linkageArray = np.arange(16)

        gfo = gfit.GlobalFitObject(tuple([a]), linkageArray)
        gfo.fit()

    def test_globfit_modelvals_same_as_indidivual(self):
        '''
            make sure that the global fit would return the same model values as the individual fitobject
        '''
        theoretical = np.loadtxt('pyplatypus/analysis/test/c_PLP0011859_q.txt')

        qvals, rvals, evals, dummy = np.hsplit(theoretical, 4)

        fitted_parameters = np.array(
            [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        a = reflect.ReflectivityFitObject(
            qvals, rvals, evals, self.coefs, fitted_parameters=fitted_parameters)
        linkageArray = np.arange(16)

        gfo = gfit.GlobalFitObject(tuple([a]), linkageArray)
        gfomodel = gfo.model(self.coefs)

        normalmodel = a.model(self.coefs)
        npt.assert_almost_equal(gfomodel, normalmodel)

    def test_globfit_modelvals_degenerate_layers(self):
        '''
            try fitting dataset with a deposited layer split into two degenerate layers
        '''
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

        fitted_parameters = np.array([6, 7, 8, 11, 12, 13, 15, 16, 17, 19])

        a = reflect.ReflectivityFitObject(
            qvals, rvals, evals, coefs, fitted_parameters=fitted_parameters)
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

        fitted_parameters = np.array([6, 7, 8, 11, 12, 13, 15, 16, 17, 19])

        a = reflect.ReflectivityFitObject(qvals,
                                          rvals,
                                          evals,
                                          coefs,
                                          fitted_parameters=fitted_parameters)
        linkageArray = np.arange(20)
        linkageArray[16] = 12
        linkageArray[17] = 15
        linkageArray[18] = 17
        linkageArray[19] = 18

        npt.assert_raises(gfit.LinkageException,
                          gfit.GlobalFitObject, tuple([a]), linkageArray)
        linkageArray[17] = 16
        linkageArray[19] = -1
        npt.assert_raises(gfit.LinkageException,
                          gfit.GlobalFitObject, tuple([a]), linkageArray)

    def test_multipledataset_corefinement(self):
        '''
            test corefinement of three datasets
        '''
        e361 = np.loadtxt('pyplatypus/analysis/test/e361r.txt')
        e365 = np.loadtxt('pyplatypus/analysis/test/e365r.txt')
        e366 = np.loadtxt('pyplatypus/analysis/test/e366r.txt')

        coefs361 = np.zeros((16))
        coefs361[0] = 2
        coefs361[1] = 1.
        coefs361[2] = 2.07
        coefs361[4] = 6.36
        coefs361[6] = 2e-5
        coefs361[7] = 3
        coefs361[8] = 10
        coefs361[9] = 3.47
        coefs361[11] = 4
        coefs361[12] = 200
        coefs361[13] = 1
        coefs361[15] = 3

        coefs365 = np.copy(coefs361)
        coefs366 = np.copy(coefs361)
        coefs365[4] = 3.47
        coefs366[4] = -0.56

        qvals361, rvals361, evals361 = np.hsplit(e361, 3)
        qvals365, rvals365, evals365 = np.hsplit(e365, 3)
        qvals366, rvals366, evals366 = np.hsplit(e366, 3)

        fitted_parameters = np.array([1, 6, 8, 12, 13])

        a = reflect.ReflectivityFitObject(
            qvals361, rvals361, evals361, coefs361, fitted_parameters=fitted_parameters)
        b = reflect.ReflectivityFitObject(
            qvals365, rvals365, evals365, coefs365, fitted_parameters=fitted_parameters)
        c = reflect.ReflectivityFitObject(
            qvals366, rvals366, evals366, coefs366, fitted_parameters=fitted_parameters)

        linkageArray = np.array(
            [[0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15],
             [16, 17, 18, 19, 20, 21, 22, 23,  8, 24,
              25,  26,  12,  27,  28,  29],
             [30, 31, 32, 33, 34, 35, 36, 37,  8, 38,  39,  40,  12,  41,  42,  43]])

        gfo = gfit.GlobalFitObject(tuple([a, b, c]), linkageArray, seed=SEED)
        np.seterr(all='ignore')
        pars, dummy, chi2 = gfo.fit()

#         modeltosave = model.Model(pars)
#         with open('pyplatypus/analysis/test/corefinee361.txt', 'w') as f:
#             modeltosave.save(f)

        with open('pyplatypus/analysis/test/corefinee361.txt', 'Ur') as f:
            savedmodel = model.Model(None, file=f)

        npt.assert_almost_equal(pars, savedmodel.parameters)
        npt.assert_almost_equal(chi2, 1718.8700218030224)

if __name__ == '__main__':
    unittest.main()
