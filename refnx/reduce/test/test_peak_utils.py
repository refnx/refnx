import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import refnx.reduce.peak_utils as peak_utils


class TestPeakUtils:
    def setup_method(self):
        pass

    def test_peak_finder(self):
        mean = 10.1234
        sd = 5.55
        x = np.linspace(-100, 100.5, 101)
        y = peak_utils.gauss(x, 0, 10, mean, sd)
        res = peak_utils.peak_finder(y, x=x)
        assert_almost_equal(res[1][0], mean)
        assert_almost_equal(res[1][1], sd)

    def test_centroid(self):
        y = np.ones(10)
        centroid, sd = peak_utils.centroid(y)
        assert_equal(centroid, 4.5)

        y = np.ones(3)
        x = np.array([0, 1.0, 9.0])
        centroid, sd = peak_utils.centroid(y, x=x)
        assert_equal(centroid, 4.5)

        y = np.ones(2)
        centroid, sd = peak_utils.centroid(y, dx=9.0)
        assert_equal(centroid, 4.5)

    def test_median(self):
        y = np.ones(10)
        median, sd = peak_utils.median(y)
        assert_equal(median, 4.5)

        y = np.ones(3)
        x = np.array([0, 1.0, 9.0])
        median, sd = peak_utils.median(y, x=x)
        assert_equal(median, 4.5)

        y = np.ones(2)
        median, sd = peak_utils.median(y, dx=9.0)
        assert_equal(median, 4.5)
