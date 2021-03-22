import refnx.util.ErrorProp as EP
import numpy as np
import os
from numpy.testing import assert_equal, assert_array_almost_equal


class TestErrorProp:
    def setup_method(self):
        pass

    def test_add(self):
        c, dc = EP.EPadd(1.1, 0.5, 1.1, 0.5)
        assert_equal(2.2, c)
        assert_equal(np.sqrt(0.5), dc)

    def test_sub(self):
        c, dc = EP.EPsub(1.1, 0.5, 1.1, 0.5)
        assert_equal(0, c)
        assert_equal(np.sqrt(0.5), dc)

    def test_mul(self):
        c, dc = EP.EPmul(1.1, 0.5, 1.1, 0.5)
        assert_array_almost_equal(1.21, c)
        assert_equal(np.sqrt(0.605), dc)

        c, dc = EP.EPmul(0, 0.5, 1.1, 0.5)
        assert_equal(0, c)
        assert_equal(0.55, dc)

    def test_div(self):
        c, dc = EP.EPdiv(1.1, 0.5, 1.1, 0.5)
        assert_array_almost_equal(1, c)
        assert_array_almost_equal(0.642824346533225, dc)

        c, dc = EP.EPdiv(0, 0.5, 1.1, 0.5)
        assert_equal(0, c)
        assert_array_almost_equal(0.45454545454545453, dc)

    def test_pow(self):
        c, dc = EP.EPpow(1.1, 0.5, 1.1, 0.5)
        assert_array_almost_equal(1.1105342410545758, c)
        assert_array_almost_equal(0.5577834505363953, dc)

        c, dc = EP.EPpow(1.1, 0.5, 1.1, 0)
        assert_array_almost_equal(1.1105342410545758, c)
        assert_array_almost_equal(0.5552671205272879, dc)

    def test_log(self):
        c, dc = EP.EPlog(1.1, 0.5)
        assert_array_almost_equal(0.09531017980432493, c)
        assert_array_almost_equal(0.45454545454545453, dc)

    def test_log10(self):
        c, dc = EP.EPlog10(1.1, 0.5)
        assert_array_almost_equal(0.04139268515822508, c)
        assert_array_almost_equal(0.19740658268329625, dc)

    def test_exp(self):
        c, dc = EP.EPexp(1.1, 0.5)
        assert_array_almost_equal(3.0041660239464334, c)
        assert_array_almost_equal(1.5020830119732167, dc)

    def test_sin(self):
        c, dc = EP.EPsin(1.1, 0.5)
        assert_array_almost_equal(0.8912073600614354, c)
        assert_array_almost_equal(0.22679806071278866, dc)

    def test_cos(self):
        c, dc = EP.EPcos(1.1, 0.5)
        assert_array_almost_equal(0.4535961214255773, c)
        assert_array_almost_equal(0.4456036800307177, dc)

    def test_tan(self):
        c, dc = EP.EPtan(1.1, 0.5)
        assert_array_almost_equal(1.9647596572486525, c)
        assert_array_almost_equal(2.430140255375921, dc)
