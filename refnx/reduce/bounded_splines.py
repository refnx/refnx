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

import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline


class BoundedUnivariateSpline(UnivariateSpline):
    r"""
    1D spline that returns a constant for x outside the specified domain.

    """

    def __init__(self, x, y, fill_value=0.0, **kwargs):
        self.bnds = [x[0], x[-1]]
        self.fill_value = fill_value
        UnivariateSpline.__init__(self, x, y, **kwargs)

    def is_outside_domain(self, x):
        x = np.asarray(x)
        return np.logical_or(x < self.bnds[0], x > self.bnds[1])

    def __call__(self, x):
        outside = self.is_outside_domain(x)

        return np.where(
            outside, self.fill_value, UnivariateSpline.__call__(self, x)
        )

    def integral(self, a, b):
        # capturing contributions outside domain of interpolation
        below_dx = np.max([0.0, self.bnds[0] - a])
        above_dx = np.max([0.0, b - self.bnds[1]])

        outside_contribution = (below_dx + above_dx) * self.fill_value

        # adjusting interval to spline domain
        a_f = np.max([a, self.bnds[0]])
        b_f = np.min([b, self.bnds[1]])

        if a_f >= b_f:
            return outside_contribution
        else:
            return outside_contribution + UnivariateSpline.integral(
                self, a_f, b_f
            )


class BoundedRectBivariateSpline(RectBivariateSpline):
    r"""
    2D spline that returns a constant for x outside the specified domain.

    Parameters
    ----------
        x : array_like
            bin edges in x direction, length m+1
        y : array_like
            bin edges in y direction, length n+1,
        z : array_like
            values of function to fit spline, m by n

    """

    def __init__(self, x, y, z, fill_value=0.0, **kwargs):
        self.xbnds = [x[0], x[-1]]
        self.ybnds = [y[0], y[-1]]
        self.fill_value = fill_value
        RectBivariateSpline.__init__(self, x, y, z, **kwargs)

    def is_outside_domain(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.logical_or(
            np.logical_or(x < self.xbnds[0], x > self.xbnds[1]),
            np.logical_or(y < self.ybnds[0], y > self.xbnds[1]),
        )

    def __call__(self, x, y):
        outside = self.is_outside_domain(x, y)

        return np.where(
            outside, self.fill_value, RectBivariateSpline.__call__(self, x, y)
        )

    def integral(self, xa, xb, ya, yb):
        assert xa <= xb
        assert ya <= yb

        total_area = (xb - xa) * (yb - ya)

        # adjusting interval to spline domain
        xa_f = np.max([xa, self.xbnds[0]])
        xb_f = np.min([xb, self.xbnds[1]])
        ya_f = np.max([ya, self.ybnds[0]])
        yb_f = np.min([yb, self.ybnds[1]])

        # Rectangle does not overlap with spline domain
        if xa_f >= xb_f or ya_f >= yb_f:
            return total_area * self.fill_value

        # Rectangle overlaps with spline domain
        else:
            spline_area = (xb_f - xa_f) * (yb_f - ya_f)
            outside_contribution = (total_area - spline_area) * self.fill_value
            return outside_contribution + RectBivariateSpline.integral(
                self, xa_f, xb_f, ya_f, yb_f
            )
