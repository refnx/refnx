import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from refnx.reflect import SLD, Slab, Structure, Spline


class TestReflect(unittest.TestCase):

    def setUp(self):
        pass

    def test_spline_smoke(self):
        # smoke test to make Spline at least gives us something
        left = SLD(1.5)(10, 3)
        right = SLD(2.5)(10, 3)
        solvent = SLD(10)(0, 3)
        a = Spline(100, [2],
                   [0.5], left, right, solvent, zgrad=False,
                   microslab_max_thickness=2)
        b = a.slabs
        assert_equal(b[:, 2], 0)
        assert_equal(b[:, 3], 0.5)

        # microslabs are assessed in the middle of the slab
        assert_equal(b[0, 1], a(0.5 * b[0, 0]))

        # with the ends turned off the profile should be a straight line
        assert_equal(a(50), 2.0)
