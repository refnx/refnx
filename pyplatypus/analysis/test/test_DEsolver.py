import unittest
import pyplatypus.analysis.DEsolver as DEsolver
import pyplatypus.analysis.model as model
import numpy as np
import numpy.testing as npt

SEED = 1


class TestDEsolver(unittest.TestCase):

    def setUp(self):
        np.seterr(invalid='raise')

    def test_differential_evolutionfit(self):
        '''
            test that the Jmin of DEsolver is the same as the function
            evaluation
        '''
        func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
        limits = np.array([[-3], [3]])
        xmin, Jmin = DEsolver.diffevol(func, limits, tol=1e-10)
        print xmin, Jmin
        npt.assert_almost_equal(Jmin, func(xmin))

    def test_select_samples(self):
        '''
            select_samples should return 5 separate random numbers.
        '''
        
        limits = np.arange(12.).reshape(2, 6)        
        solver = DEsolver.DEsolver(None, limits, popsize=1)
        candidate = 0
        r1, r2, r3, r4, r5 = solver.select_samples(candidate, 1, 1, 1, 1, 1)
        assert len(np.unique(np.array([candidate, r1, r2, r3, r4, r5]))) == 6

if __name__ == '__main__':
    unittest.main()
