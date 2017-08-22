"""
Testing rebin histogram values.
"""

import numpy as np
from numpy.random import uniform
from numpy.testing import (assert_allclose, assert_equal)
import uncertainties.unumpy as unp
import refnx.reduce.rebin as rebin


class TestRebin(object):
    def setup_method(self):
        pass

    # -----------------------------------------------------------------------#
    #  Tests for piecewise continuous rebinning
    # -----------------------------------------------------------------------#
    def test_x2_same_as_x1(self):
        """
        x2 same as x1
        """
        # old size
        m = 6

        # new size
        n = 6

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(0., 1., n + 1)

        # some arbitrary distribution
        y_old = 1. + np.sin(x_old[:-1] * np.pi) / np.ediff1d(x_old)

        # rebin
        y_new = rebin.rebin(x_old, y_old, x_new)

        assert_allclose(y_new, y_old)

    def test_x2_surrounds_x1(self):
        """
        x2 range surrounds x1 range
        """
        # old size
        m = 2

        # new size
        n = 3

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(-0.1, 1.2, n + 1)

        # some arbitrary distribution
        y_old = 1. + np.sin(x_old[:-1] * np.pi) / np.ediff1d(x_old)

        # rebin
        y_new = rebin.rebin(x_old, y_old, x_new)

        # compute answer here to check rebin
        y_old_ave = y_old / np.ediff1d(x_old)
        y_new_here = [y_old_ave[0] * (x_new[1] - 0.),
                      y_old_ave[0] * (x_old[1] - x_new[1]) +
                      y_old_ave[1] * (x_new[2] - x_old[1]),
                      y_old_ave[1] * (x_old[-1] - x_new[-2])]

        assert_allclose(y_new, y_new_here)
        assert_allclose(y_new.sum(), y_old.sum())

    def test_x2_surrounds_x1_2(self):
        """
        x2 has some bins that span several x1 bins
        Also tests uncertainty propagation. Values calculated using
        original jhykes piecewise constant code
        """
        # old size
        m = 10

        # new size
        n = 3

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(-0.1, 1.2, n + 1)

        # some arbitrary distribution
        y_old = 1. + np.sin(x_old[:-1] * np.pi) / np.ediff1d(x_old)

        # with uncertainties
        np.random.seed(1)
        y_old_sd = 0.1 * y_old * uniform((m,))

        # rebin
        y_new, y_new_sd = rebin.rebin(x_old,
                                      y_old,
                                      x_new,
                                      y1_sd=y_old_sd)

        # compute answer here to check rebin
        y_new_here = np.array([14.99807911, 44.14135692, 13.99807911])
        y_new_here_sd = np.array([5.381524308729351,
                                  12.73174109312833,
                                  5.345145324353735])

        assert_allclose(y_new, y_new_here)
        assert_allclose(y_new_sd, y_new_here_sd)
        assert_allclose(y_new.sum(), y_old.sum())

    def test_x2_lower_than_x1(self):
        """
        x2 range is completely lower than x1 range
        """
        # old size
        m = 2

        # new size
        n = 3

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(-0.2, -0.0, n + 1)

        # some arbitrary distribution
        y_old = 1. + np.sin(x_old[:-1] * np.pi) / np.ediff1d(x_old)

        # rebin
        y_new = rebin.rebin(x_old, y_old, x_new)

        assert_allclose(y_new, [0., 0., 0.])
        assert_allclose(y_new.sum(), 0.)

    def test_x2_above_x1(self):
        """
        x2 range is completely above x1 range
        """
        # old size
        m = 20

        # new size
        n = 30

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(1.2, 10., n + 1)

        # some arbitrary distribution
        y_old = 1. + np.sin(x_old[:-1] * np.pi) / np.ediff1d(x_old)

        # rebin
        y_new = rebin.rebin(x_old, y_old, x_new)

        assert_allclose(y_new, np.zeros((n,)))
        assert_allclose(y_new.sum(), 0.)

    def test_x2_in_x1(self):
        """
        x2 only has one bin, and it is surrounded by x1 range
        """
        # old size
        m = 4

        # new size
        n = 1

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(0.3, 0.65, n + 1)

        # some arbitrary distribution
        y_old = 1. + np.sin(x_old[:-1] * np.pi) / np.ediff1d(x_old)

        # rebin
        y_new = rebin.rebin(x_old, y_old, x_new)

        # compute answer here to check rebin
        y_old_ave = y_old / np.ediff1d(x_old)
        y_new_here = (y_old_ave[1] * (x_old[2] - x_new[0]) +
                      y_old_ave[2] * (x_new[1] - x_old[2]))

        assert_allclose(y_new, y_new_here)

    def test_y1_uncertainties(self):
        """
        x2 range surrounds x1 range, y1 has uncertainties
        """
        # old size
        m = 2

        # new size
        n = 3

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(-0.1, 1.2, n + 1)

        # some arbitrary distribution
        y_old = 1. + np.sin(x_old[:-1] * np.pi) / np.ediff1d(x_old)

        # with uncertainties
        y_old = unp.uarray(y_old, 0.1 * y_old * uniform((m,)))

        # rebin
        y_new, y_new_sd = rebin.rebin(x_old,
                                      unp.nominal_values(y_old),
                                      x_new,
                                      y1_sd=unp.std_devs(y_old))

        # compute answer here to check rebin
        y_old_ave = y_old / np.ediff1d(x_old)
        y_new_here = np.array(
            [y_old_ave[0] * (x_new[1] - 0.),
             y_old_ave[0] * (x_old[1] - x_new[1]) +
             y_old_ave[1] * (x_new[2] - x_old[1]),
             y_old_ave[1] * (x_old[-1] - x_new[-2])])

        # mean or nominal value comparison
        assert_allclose(y_new,
                        unp.nominal_values(y_new_here))

        # mean or nominal value comparison
        assert_allclose(y_new_sd,
                        unp.std_devs(y_new_here))

        assert_allclose(y_new.sum(),
                        unp.nominal_values(y_new_here).sum())

    def test_rebinND(self):
        input = np.arange(24).reshape(4, 3, 2)
        x1 = np.arange(5)
        x2 = np.array([0, 1, 2.5, 4])
        y1 = np.arange(4)
        y2 = np.array([0.5, 1.5])
        output = rebin.rebinND(input, (0, 1), (x1, y1), (x2, y2))
        res = np.array([[[1., 2.]],
                        [[13.5, 15.]],
                        [[25.5, 27.]]])

        assert_equal(res, output)
