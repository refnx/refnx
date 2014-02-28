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


if __name__ == '__main__':
    unittest.main()
