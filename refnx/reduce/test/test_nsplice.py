__author__ = 'anz'
import unittest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_)
import os.path
import refnx.reduce.nsplice as nsplice

path = os.path.dirname(os.path.abspath(__file__))


class TestNSplice(unittest.TestCase):

    def setUp(self):
        fname0 = os.path.join(path, 'PLP0000708.dat')
        self.qv0, self.rv0, self.drv0 = np.loadtxt(fname0,
                                                  usecols=(0, 1, 2),
                                                  unpack=True,
                                                  skiprows=1)

        fname1 = os.path.join(path, 'PLP0000709.dat')
        self.qv1, self.rv1, self.drv1 = np.loadtxt(fname1,
                                                  usecols=(0, 1, 2),
                                                  unpack=True,
                                                  skiprows=1)
        self.scale = 0.1697804743373886
        self.dscale = 0.0018856527316943896

    def test_nsplice(self):
        # test splicing of two datasets
        scale, dscale = nsplice.get_scaling_in_overlap(self.qv0,
                                                       self.rv0,
                                                       self.drv0,
                                                       self.qv1,
                                                       self.rv1,
                                                       self.drv1)
        assert_almost_equal(scale, self.scale)
        assert_almost_equal(dscale, self.dscale)

    def test_nsplice_nooverlap(self):
        # test splicing of two datasets if there's no overlap
        # the scale factor should be np.nan
        qv0 = self.qv0[0:10]
        scale, dscale = nsplice.get_scaling_in_overlap(qv0,
                                                       self.rv0,
                                                       self.drv0,
                                                       self.qv1,
                                                       self.rv1,
                                                       self.drv1)
        assert_(not np.isfinite(scale))
        assert_(not np.isfinite(dscale))

    def test_nsplice_notsorted(self):
        # test splicing of two datasets if there's no sorting
        # of the data
        vec0 = np.arange(np.size(self.qv0))
        np.random.shuffle(vec0)
        vec1 = np.arange(np.size(self.qv1))
        np.random.shuffle(vec1)

        scale, dscale = nsplice.get_scaling_in_overlap(self.qv0[vec0],
                                                       self.rv0[vec0],
                                                       self.drv0[vec0],
                                                       self.qv1[vec1],
                                                       self.rv1[vec1],
                                                       self.drv1[vec1])
        assert_almost_equal(scale, self.scale)
        assert_almost_equal(dscale, self.dscale)


if __name__ == '__main__':
    unittest.main()