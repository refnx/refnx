import unittest
import refnx.reduce.platypusnexus as plp
import numpy as np
import os
from refnx.reduce import reduce_stitch_files
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
                           assert_array_less)


class TestPlatypusNexus(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(__file__)
        self.path = path

        return 0

    def test_smoke(self):
        # a quick smoke test to check that the reduction can occur
        a = reduce_stitch_files([708, 709, 710], [711, 711, 711], rebin_percent=2)
        a.save('test1.dat')