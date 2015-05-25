"""
Copyright (c) 2011, Joshua M. Hykes
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL JOSHUA M. HYKES BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
Testing rebin histogram values.
"""

import numpy as np
import unittest
from numpy.random import uniform
from numpy.testing import (assert_allclose, assert_equal)
from scipy.interpolate import splrep, splint

import uncertainties.unumpy as unp

import refnx.reduce.rebin as rebin
from refnx.reduce.bounded_splines import (BoundedUnivariateSpline,
                                          BoundedRectBivariateSpline)


class TestRebin(unittest.TestCase):
    def setUp(self):
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
        y_new = rebin.rebin(x_old, y_old, x_new,
                            interp_kind='piecewise_constant')

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
        y_new = rebin.rebin(x_old, y_old, x_new,
                            interp_kind='piecewise_constant')

        # compute answer here to check rebin
        y_old_ave  = y_old / np.ediff1d(x_old)
        y_new_here = [y_old_ave[0] * (x_new[1] - 0.),
                      y_old_ave[0] * (x_old[1]- x_new[1])
                      + y_old_ave[1] * (x_new[2] - x_old[1]),
                      y_old_ave[1] * (x_old[-1] - x_new[-2])]

        assert_allclose(y_new, y_new_here)
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
        y_new = rebin.rebin(x_old, y_old, x_new,
                            interp_kind='piecewise_constant')

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
        y_new = rebin.rebin(x_old, y_old, x_new,
                            interp_kind='piecewise_constant')

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
        y_new = rebin.rebin(x_old, y_old, x_new,
                            interp_kind='piecewise_constant')

        # compute answer here to check rebin
        y_old_ave  = y_old / np.ediff1d(x_old)
        y_new_here = (y_old_ave[1] * (x_old[2] - x_new[0])
                      + y_old_ave[2] * (x_new[1] - x_old[2]) )

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
        y_new = rebin.rebin(x_old, y_old, x_new,
                            interp_kind='piecewise_constant')

        # compute answer here to check rebin
        y_old_ave  = y_old / np.ediff1d(x_old)
        y_new_here = np.array(
                     [y_old_ave[0] * (x_new[1] - 0.),
                      y_old_ave[0] * (x_old[1] - x_new[1])
                      + y_old_ave[1]*(x_new[2] - x_old[1]),
                      y_old_ave[1] * (x_old[-1] - x_new[-2])])


        # mean or nominal value comparison
        assert_allclose(unp.nominal_values(y_new),
                        unp.nominal_values(y_new_here))

        # mean or nominal value comparison
        assert_allclose(unp.std_devs(y_new),
                        unp.std_devs(y_new_here))
        assert_allclose(unp.nominal_values(y_new).sum(),
                        unp.nominal_values(y_new_here).sum())


    # -----------------------------------------------------------------------#
    #  Tests for cubic-spline rebinning
    # -----------------------------------------------------------------------#
    def test_x2_surrounds_x1_with_constant_distribution(self):
        """
        x2 domain completely surrounds x1 domain
        """
        # old size
        m = 20

        # new size
        n = 30

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(-0.5, 1.5, n + 1)

        # constant spline
        mms_spline = BoundedUnivariateSpline([0, .1, .2, 1],
                                             [1, 1, 1, 1], s=0.)

        y_old = np.array([mms_spline.integral(x_old[i],
                                              x_old[i + 1]) for i in range(m)])

        y_new_mms = np.array([mms_spline.integral(x_new[i],
                                                  x_new[i + 1])
                              for i in range(n)])


        # rebin
        y_new = rebin.rebin(x_old, y_old, x_new, interp_kind=3)

        assert_allclose(y_new, y_new_mms)


    def test_x2_left_overlap_x1_with_constant_distribution(self):
        """
        x2 domain overlaps x1 domain from the left
        """
        # old size
        m = 20

        # new size
        n = 30

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(-0.75, 0.45, n + 1)

        # constant spline
        mms_spline = BoundedUnivariateSpline([0, .1, .2, 1],
                                             [1, 1, 1, 1], s=0.)

        y_old = np.array([mms_spline.integral(x_old[i], x_old[i + 1])
                          for i in range(m) ])

        y_new_mms = np.array([mms_spline.integral(x_new[i],
                                                   x_new[i+1])
                              for i in range(n)])

        # rebin
        y_new = rebin.rebin(x_old, y_old, x_new, interp_kind=3)

        assert_allclose(y_new, y_new_mms)


    def test_x2_right_overlap_x1_with_constant_distribution(self):
        """
        x2 domain overlaps x1 domain from the right
        """
        # old size
        m = 20

        # new size
        n = 30

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(0.95, 1.05, n + 1)

        # constant spline
        mms_spline = BoundedUnivariateSpline([0, .1, .2, 1],
                                             [1, 1, 1, 1], s=0.)

        y_old = np.array([mms_spline.integral(x_old[i],
                                              x_old[i + 1])
                          for i in range(m)])

        y_new_mms = np.array([mms_spline.integral(x_new[i],
                                                  x_new[i + 1])
                              for i in range(n)])

        # rebin
        y_new = rebin.rebin(x_old, y_old, x_new, interp_kind=3)
        assert_allclose(y_new, y_new_mms, rtol=1e-6, atol=1e-4)


    def test_x1_surrounds_x2_with_constant_distribution(self):
        """
        x1 domain surrounds x2
        """
        # old size
        m = 20

        # new size
        n = 30

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.linspace(0.05, 0.26, n + 1)

        # constant spline
        mms_spline = BoundedUnivariateSpline([0, .1, .2, 1],
                                             [1, 1, 1, 1], s=0.)

        y_old = np.array([mms_spline.integral(x_old[i],
                                               x_old[i + 1])
                          for i in range(m)])

        y_new_mms = np.array([mms_spline.integral(x_new[i],
                                                  x_new[i + 1])
                              for i in range(n)])

        # rebin
        y_new = rebin.rebin(x_old, y_old, x_new, interp_kind=3)

        assert_allclose(y_new, y_new_mms)


    def test_x2_surrounds_x1_sine_spline(self):
        """
        x2 range is completely above x1 range
        using a random vector to build spline
        """
        # old size
        m = 5

        # new size
        n = 6

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.array([-.3, -.09, 0.11, 0.14, 0.2, 0.28, 0.73])

        subbins = np.array([-.3, -.09, 0., 0.11, 0.14, 0.2, 0.28, 0.4, 0.6,
                            0.73])

        y_old = 1. + np.sin(x_old[:-1] * np.pi)

        # compute spline ----------------------------------
        x_mids = x_old[:-1] + 0.5 * np.ediff1d(x_old)
        xx = np.hstack([x_old[0], x_mids, x_old[-1]])
        yy = np.hstack([y_old[0], y_old, y_old[-1]])

        # build spline
        spl = splrep(xx, yy)

        area_old = np.array([splint(x_old[i], x_old[i + 1], spl)
                             for i in range(m)])

        # computing subbin areas
        area_subbins = np.zeros((subbins.size - 1,))
        for i in range(area_subbins.size):
            a, b = subbins[i: i + 2]
            a = max([a, x_old[0]])
            b = min([b, x_old[-1]])
            if b > a:
                area_subbins[i] = splint(a, b, spl)

        # summing subbin contributions in y_new_ref
        y_new_ref = np.zeros((x_new.size - 1,))
        y_new_ref[1] = y_old[0] * area_subbins[2] / area_old[0]
        y_new_ref[2] = y_old[0] * area_subbins[3] / area_old[0]
        y_new_ref[3] = y_old[0] * area_subbins[4] / area_old[0]
        y_new_ref[4] = y_old[1] * area_subbins[5] / area_old[1]

        y_new_ref[5]  = y_old[1] * area_subbins[6] / area_old[1]
        y_new_ref[5] += y_old[2] * area_subbins[7] / area_old[2]
        y_new_ref[5] += y_old[3] * area_subbins[8] / area_old[3]

        # call rebin function
        y_new = rebin.rebin(x_old, y_old, x_new, interp_kind=3)

        assert_allclose(y_new, y_new_ref)


    def test_y1_uncertainties_spline_with_constant_distribution(self):
        """

        """
        # old size
        m = 5

        # new size
        n = 6

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        x_new = np.array([-.3, -.09, 0.11, 0.14, 0.2, 0.28, 0.73])

        subbins = np.array([-.3, -.09, 0., 0.11, 0.14, 0.2, 0.28, 0.4, 0.6,
                            0.73])

        y_old = 1. + np.sin(x_old[:-1] * np.pi)

        # compute spline ----------------------------------
        x_mids = x_old[:-1] + 0.5 * np.ediff1d(x_old)
        xx = np.hstack([x_old[0], x_mids, x_old[-1]])
        yy = np.hstack([y_old[0], y_old, y_old[-1]])

        # build spline
        spl = splrep(xx, yy)

        area_old = np.array([splint(x_old[i],x_old[i+1], spl)
                             for i in range(m)])

        # with uncertainties
        y_old = unp.uarray(y_old, 0.1 * y_old * uniform((m,)))

        # computing subbin areas
        area_subbins = np.zeros((subbins.size - 1,))
        for i in range(area_subbins.size):
            a, b = subbins[i: i + 2]
            a = max([a, x_old[0]])
            b = min([b, x_old[-1]])
            if b > a:
                area_subbins[i] = splint(a, b, spl)

        # summing subbin contributions in y_new_ref
        a = np.zeros((x_new.size - 1,))
        y_new_ref = unp.uarray(a, a)
        y_new_ref[1] = y_old[0] * area_subbins[2] / area_old[0]
        y_new_ref[2] = y_old[0] * area_subbins[3] / area_old[0]
        y_new_ref[3] = y_old[0] * area_subbins[4] / area_old[0]
        y_new_ref[4] = y_old[1] * area_subbins[5] / area_old[1]

        y_new_ref[5]  = y_old[1] * area_subbins[6] / area_old[1]
        y_new_ref[5] += y_old[2] * area_subbins[7] / area_old[2]
        y_new_ref[5] += y_old[3] * area_subbins[8] / area_old[3]

        # call rebin function
        y_new = rebin.rebin(x_old, y_old, x_new, interp_kind=3)

        # mean or nominal value comparison
        assert_allclose(unp.nominal_values(y_new),
                        unp.nominal_values(y_new_ref))

        # mean or nominal value comparison
        assert_allclose(unp.std_devs(y_new),
                        unp.std_devs(y_new_ref))

    # ---------------------------------------------------------------------------- #
    #  Tests for 2d rebinning
    # ---------------------------------------------------------------------------- #

    def test_2d_same(self):
        """
        x1, y1 == x2, y2 implies z1 == z2
        2d
        """
        # old size
        m = 20
        n = 30

        # bin edges
        x_old = np.linspace(0., 1., m + 1)
        y_old = np.linspace(-0.5, 1.5, n + 1)

        z_old = np.random.random((m, n))

        # rebin
        z_new = rebin.rebin2d(x_old, y_old, z_old, x_old, y_old)

        assert_allclose(z_old, z_new)

    def test_2d_constant_distribution(self):
        """
        various new domains with a constant underlying distribution
        2d
        """
        # old size
        m = 8
        n = 11

        # new size
        p = 5
        q = 14

        new_bounds = [ (0., 1., -1.5, 1.7),
                       (0., 1., -1.5, 0.7),
                       (0., 1., -1.5, -0.7),
                       (-1., 1.5, -1.5, 1.7),
                       (-1., 0.5, -1., 0.5),
                       (0.1, 0.6, 0.1, 0.5),
                       (0.01, 0.02, -10.0, 20.7)]

        for (a, b, c, d) in new_bounds:

            # bin edges
            x_old = np.linspace(0., 1., m + 1)
            y_old = np.linspace(-0.5, 1.5, n + 1)

            x_new = np.linspace(a, b, p + 1)
            y_new = np.linspace(c, d, q + 1)

            # constant spline
            z_old = np.ones((m + 1, n + 1))
            mms_spline = BoundedRectBivariateSpline(x_old, y_old, z_old, s=0.)

            z_old = np.zeros((m,n))
            for i in range(m):
                for j in range(n):
                    z_old[i, j] =  mms_spline.integral(x_old[i], x_old[i + 1],
                                                       y_old[j], y_old[j + 1])

            z_new_mms = np.zeros((p, q))
            for i in range(p):
                for j in range(q):
                    z_new_mms[i, j] =  mms_spline.integral(x_new[i],
                                                           x_new[i + 1],
                                                           y_new[j],
                                                           y_new[j + 1])

            # rebin
            z_new = rebin.rebin2d(x_old, y_old, z_old, x_new, y_new)

            assert_allclose(z_new, z_new_mms)

    def test_rebin_along_axis(self):
        y = np.arange(100).reshape(20, 5)
        x1 = np.arange(21)
        x2  =np.array([0,1,2.5,5,20])
        y2 = rebin.rebin_along_axis(y, x1, x2)

        res = np.array([[   0. ,    1. ,    2. ,    3. ,    4.],
                        [  10. ,   11.5,   13. ,   14.5,   16.],
                        [  40. ,   42.5,   45. ,   47.5,   50.],
                        [ 900. ,  915. ,  930. ,  945. ,  960.]])
        assert_equal(y2, res)

        y = np.arange(24).reshape(4, 3, 2)
        x1 = np.arange(5)
        x2  =np.array([0,1,2.5,4])
        y2 = rebin.rebin_along_axis(y, x1, x2)
        res = np.array([[[  0. ,   1. ],
                         [  2. ,   3. ],
                         [  4. ,   5. ]],

                        [[ 12. ,  13.5],
                         [ 15. ,  16.5],
                         [ 18. ,  19.5]],

                        [[ 24. ,  25.5],
                         [ 27. ,  28.5],
                         [ 30. ,  31.5]]])
        assert_equal(res, y2)

    def test_rebinND(self):
        input = np.arange(24).reshape(4, 3, 2)
        x1 = np.arange(5)
        x2  =np.array([0,1,2.5,4])
        y1 = np.arange(4)
        y2 = np.array([0.5, 1.5])
        output = rebin.rebinND(input, (0, 1), (x1, y1), (x2, y2))
        res = np.array([[[  1. ,   2. ]],
                        [[ 13.5,  15. ]],
                        [[ 25.5,  27. ]]])

        assert_equal(res, output)

if __name__ == '__main__':
    unittest.main()