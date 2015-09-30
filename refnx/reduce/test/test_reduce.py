import unittest
import numpy as np
import os.path
from refnx.reduce import reduce_stitch, ReducePlatypus
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
                           assert_array_less)


class TestReduce(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(__file__)
        self.path = path

        return 0

    def test_smoke(self):
        # a quick smoke test to check that the reduction can occur
        a, fname = reduce_stitch([708, 709, 710], [711, 711, 711],
                          data_folder=self.path, rebin_percent=2)
        a.save('test1.dat')

    def test_reduction_method(self):
        # a quick smoke test to check that the reduction can occur
        a = ReducePlatypus('PLP0000711.nx.hdf', data_folder=self.path,
                           rebin_percent=4)

        # try reduction with the reduce method
        a.reduce('PLP0000708.nx.hdf', data_folder=self.path, rebin_percent=4)

        # try reduction with the __call__ method
        a('PLP0000708.nx.hdf', data_folder=self.path, rebin_percent=4)

        # try writing offspecular data
        a.write_offspecular('offspec.xml', 0)

    def test_event_reduction(self):
        # check that eventmode reduction can occur, and that there are the
        # correct number of datasets produced.
        a = ReducePlatypus(
            os.path.join(self.path, 'PLP0011613.nx.hdf'),
            reflect=os.path.join(self.path, 'PLP0011641.nx.hdf'),
            integrate=0, rebin_percent=2,
            eventmode=[0, 900, 1800])
        assert_equal(a.ydata.shape[0], 2)


if __name__ == '__main__':
    unittest.main()