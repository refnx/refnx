from __future__ import division
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
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

"""
Rebin 1D and 2D histograms.

"""
import numpy as np

try:
    import uncertainties.unumpy as unp
    nom = unp.nominal_values
except ImportError:
    nom = lambda x: x

from .bounded_splines import BoundedUnivariateSpline, BoundedRectBivariateSpline


def midpoints(xx):
    """Return midpoints of edges in xx."""
    return xx[:-1] + 0.5*np.ediff1d(xx)


def edge_step(x, y, **kwargs):
    """
    Plot a histogram with edges and bin values precomputed. The normal
    matplotlib hist function computes the bin values internally.

    Input
    -----
     * x : n+1 array of bin edges.
     * y : n array of histogram values.

    """
    return plt.plot(x, np.hstack([y,y[-1]]), drawstyle='steps-post', **kwargs)


def rebin(x1, y1, x2, interp_kind=3):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.

    Input
    -----
     * x1 : m+1 array of old bin edges.
     * y1 : m array of old histogram values. This is the total number in
              each bin, not an average.
     * x2 : n+1 array of new bin edges.
     * interp_kind : how is the underlying unknown continuous distribution
                      assumed to look: {3, 'piecewise_constant'}
                      3 is cubic splines
                      piecewise_constant is constant in each histogram bin


    Returns
    -------
     * y2 : n array of rebinned histogram values.

    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """

    if interp_kind == 'piecewise_constant':
        return rebin_piecewise_constant(x1, y1, x2)
    else:
        return rebin_spline(x1, y1, x2, interp_kind=interp_kind)


def rebin_spline(x1, y1, x2, interp_kind):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.

    Input
    -----
     * x1 : m+1 array of old bin edges.
     * y1 : m array of old histogram values. This is the total number in
              each bin, not an average.
     * x2 : n+1 array of new bin edges.
     * interp_kind : how is the underlying unknown continuous distribution
                      assumed to look: {'cubic'}

    Returns
    -------
     * y2 : n array of rebinned histogram values.

    The cubic spline fit (which is the only interp_kind tested)
    uses the UnivariateSpline class from Scipy, which uses FITPACK.
    The boundary condition used is not-a-knot, where the second and
    second-to-last nodes are not included as knots (but they are still
    interpolated).

    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """
    m = y1.size
    n = x2.size - 1

    # midpoints of x1
    x1_mid = midpoints(x1)

    # constructing data for spline
    #  To get the spline to flatten out at the edges, duplicate bin mid values
    #   as value on the two boundaries.
    xx = np.hstack([x1[0], x1_mid, x1[-1]])
    yy = np.hstack([y1[0], y1, y1[-1]])

    # strip uncertainties from data
    yy = nom(yy)

    # instantiate spline, s=0 gives interpolating spline
    spline = BoundedUnivariateSpline(xx, yy, s=0., k=interp_kind)

    # area under spline for each old bin
    areas1 = np.array([spline.integral(x1[i], x1[i+1]) for i in range(m)])


    # insert old bin edges into new edges
    x1_in_x2 = x1[ np.logical_and(x1 > x2[0], x1 < x2[-1]) ]
    indices  = np.searchsorted(x2, x1_in_x2)
    subbin_edges = np.insert(x2, indices, x1_in_x2)

    # integrate over each subbin
    subbin_areas = np.array([spline.integral(subbin_edges[i],
                                             subbin_edges[i+1])
                              for i in range(subbin_edges.size-1)])

    # make subbin-to-old bin map
    subbin_mid = midpoints(subbin_edges)
    sub2old = np.searchsorted(x1, subbin_mid) - 1

    # make subbin-to-new bin map
    sub2new = np.searchsorted(x2, subbin_mid) - 1

    # loop over subbins
    y2 = [0. for i in range(n)]
    for i in range(subbin_mid.size):
        # skip subcells which don't lie in range of x1
        if sub2old[i] == -1 or sub2old[i] == x1.size-1:
            continue
        else:
            y2[sub2new[i]] += ( y1[sub2old[i]] * subbin_areas[i]
                                               / areas1[sub2old[i]] )

    return np.array(y2)


def rebin_piecewise_constant(x1, y1, x2):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.

    Input
    -----
     * x1 : m+1 array of old bin edges.
     * y1 : m array of old histogram values. This is the total number in
              each bin, not an average.
     * x2 : n+1 array of new bin edges.

    Returns
    -------
     * y2 : n array of rebinned histogram values.

    The rebinning algorithm assumes that the counts in each old bin are
    uniformly distributed in that bin.

    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """

    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)

    # Divide y1 by bin widths.
    #  This converts y-values from bin total to bin average over bin width.
    x1_bin_widths = np.ediff1d(x1)
    y1_ave = y1 / x1_bin_widths

    # allocating y2 vector
    n  = x2.size - 1
    y2 = []

    # loop over all new bins
    for i in range(n):
        x2_lo, x2_hi = x2[i], x2[i+1]

        i_lo, i_hi = np.searchsorted(x1, [x2_lo, x2_hi])

        # new bin out of x1 range
        if i_hi == 0 or i_lo == x1.size:
            y2.append( 0. )
            continue

        # new bin totally covers x1 range
        elif i_lo == 0 and i_hi == x1.size:
            sub_edges = x1
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave

        # new bin overlaps lower x1 boundary
        elif i_lo == 0:
            sub_edges = np.hstack( [ x1[i_lo:i_hi], x2_hi ] )
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave[i_lo:i_hi]

        # new bin overlaps upper x1 boundary
        elif i_hi == x1.size:
            sub_edges = np.hstack( [ x2_lo, x1[i_lo:i_hi] ] )
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave[i_lo-1:i_hi]

        # new bin is enclosed in x1 range
        else:
            sub_edges = np.hstack( [ x2_lo, x1[i_lo:i_hi], x2_hi ] )
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave[i_lo-1:i_hi]

        y2.append( (sub_dx * sub_y_ave).sum() )

    return np.array(y2)


def rebin2d(x1, y1, z1, x2, y2, interp_kind=3):
    """
    Rebin 2d histogram values z1 from old rectangular bin
    edges x1, y1 to new edges x2, y2.

    Input
    -----
     * x1 : m+1 array of old bin x edges.
     * y1 : n+1 array of old bin y edges.
     * z1 : m-by-n array of old histogram values. This is the total number in
              each bin, not an average.
     * x2 : p+1 array of new bin x edges.
     * x2 : q+1 array of new bin y edges.
     * interp_kind : how is the underlying unknown continuous distribution
                      assumed to look: {3}
                      3 - bivariate cubic spline

    Returns
    -------
     * z2 : p-by-q array of rebinned histogram values.

    The cubic spline fit (which is the only interp_kind tested)
    uses the BivariateSpline class from Scipy, which uses FITPACK.
    The boundary condition used is not-a-knot, where the second and
    second-to-last nodes are not included as knots (but they are still
    interpolated).

    Bins in x2 x y2 that are entirely outside the range of x1 x y1
    are assigned 0.
    """
    m, n = z1.shape
    assert x1.size == m+1
    assert y1.size == n+1

    p = x2.size - 1
    q = y2.size - 1

    # midpoints of x1
    x1_mid = midpoints(x1)
    y1_mid = midpoints(y1)

    # constructing data for spline
    #  To get the spline to flatten out at the edges, duplicate bin mid values
    #   on the interpolation boundaries.
    xx = np.hstack([x1[0], x1_mid, x1[-1]])
    yy = np.hstack([y1[0], y1_mid, y1[-1]])

    c1 = np.hstack([z1[0,0], z1[:,0], z1[-1,0]])
    c2 = np.vstack([z1[0,:], z1, z1[-1,:]])
    c3 = np.hstack([z1[0,-1], z1[:,-1], z1[-1,-1]])

    zz = np.hstack([c1[:,np.newaxis], c2, c3[:,np.newaxis]])

    zz = nom(zz)

    # instantiate spline, s=0 gives interpolating spline
    spline = BoundedRectBivariateSpline(xx, yy, zz, s=0.,
                                        kx=interp_kind,
                                        ky=interp_kind)

    # area under spline for each old bin
    # todo: only integrate over old bins which will contribute to new bins
    areas1 = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            areas1[i,j] = spline.integral(x1[i], x1[i+1], y1[j], y1[j+1])


    # insert old bin edges into new edges
    #  into x
    x1_in_x2 = x1[ np.logical_and(x1 > x2[0], x1 < x2[-1]) ]
    x_indices  = np.searchsorted(x2, x1_in_x2)
    subbin_xedges = np.insert(x2, x_indices, x1_in_x2)

    #  into y
    y1_in_y2 = y1[ np.logical_and(y1 > y2[0], y1 < y2[-1]) ]
    y_indices  = np.searchsorted(y2, y1_in_y2)
    subbin_yedges = np.insert(y2, y_indices, y1_in_y2)

    # integrate over each subbin
    ms = subbin_xedges.size-1
    ns = subbin_yedges.size-1
    subbin_areas = np.zeros((ms,ns))
    for i in range(ms):
        for j in range(ns):
            subbin_areas[i,j] = spline.integral(
                                    subbin_xedges[i], subbin_xedges[i+1],
                                    subbin_yedges[j], subbin_yedges[j+1],
                                               )


    # make subbin-to-old bin map
    subbin_xmid = midpoints(subbin_xedges)
    x_sub2old = np.searchsorted(x1, subbin_xmid) - 1

    subbin_ymid = midpoints(subbin_yedges)
    y_sub2old = np.searchsorted(y1, subbin_ymid) - 1

    # make subbin-to-new bin map
    x_sub2new = np.searchsorted(x2, subbin_xmid) - 1
    y_sub2new = np.searchsorted(y2, subbin_ymid) - 1

    # loop over subbins
    z2 = [[0. for i in range(q)] for j in range(p)]
    for i in range(ms):
        for j in range(ns):
            # skip subcells which don't lie in range of x1 or y1
            if ( x_sub2old[i] == -1 or x_sub2old[i] == m or
                 y_sub2old[j] == -1 or y_sub2old[j] == n ):
                continue
            else:
                z2[x_sub2new[i]][y_sub2new[j]] += (
                         z1[x_sub2old[i],y_sub2old[j]]
                         * subbin_areas[i,j] /
                         areas1[x_sub2old[i], y_sub2old[j]] )

    return np.array(z2)


if __name__ == '__main__':
    # demo rebin() ---------------------------------------------------

    # old size
    m = 18

    # new size
    n = 30

    # bin edges
    x_old = np.linspace(0., 1., m+1)
    x_new = np.linspace(-0.01, 1.02, n+1)

    # some arbitrary distribution
    y_old = np.sin(x_old[:-1]*np.pi)

    # rebin
    y_new = rebin(x_old, y_old, x_new)

    # plot results ----------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure()
    edge_step(x_old, y_old, label='old')
    edge_step(x_new, y_new, label='new')

    plt.legend()
    plt.title("bin totals -- new is lower because its bins are narrower")

    plt.show()
