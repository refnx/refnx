from __future__ import division
import numpy as np

"""
Rebins histograms in a piecewise-constant fashion.  Original code based
on Joshua Hykes code.  However, the error propagation (with the uncertainties
package) was _very_ slow, so I've extracted the piecewise-constant code and
modified it.
"""

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

def rebin_along_axis(y1, x1, x2, axis=0, y1_sd=None):
    """
    Rebins an N-dimensional array along a given axis, in a piecewise-constant
    fashion.

    Parameters
    ----------
    y1 : array_like
        The input image
    x1 : array_like
        The monotonically increasing/decreasing original bin edges along
        `axis`, must be 1 greater than `np.size(y1, axis)`.
    y2 : array_like
        The final bin_edges along `axis`.
    axis : int
        The axis to be rebinned, it must exist in the original image.
    y1_sd : array_like, optional
        Standard deviations for each pixel in y1.

    Returns
    -------
    output : np.ndarray
    --OR--
    output, output_sd : np.ndarray
        The rebinned image.
    """

    orig_shape = np.array(y1.shape)
    num_axes = np.size(orig_shape)

    # Output is going to need reshaping
    new_shape = np.copy(orig_shape)
    new_shape[axis] = np.size(x2) - 1

    if axis > num_axes - 1:
        raise ValueError("That axis is not in y1")

    if np.size(y1, axis) != np.size(x1) - 1:
        raise ValueError("The original number of xbins does not match the axis"
                         "size")

    odtype = np.dtype('float')
    if y1.dtype is np.dtype('O'):
        odtype = np.dtype('O')

    output = np.empty(new_shape, dtype=odtype)
    output_sd = np.copy(output)

    it = np.nditer(y1, flags=['multi_index', 'refs_ok'])
    it.remove_axis(axis)

    while not it.finished:
        a = list(it.multi_index)
        a.insert(axis, slice(None))
        if y1_sd is not None:
            rebinned, rebinned_sd = rebin(x1,
                                          y1[a],
                                          x2,
                                          y1_sd=y1_sd[a])

            output_sd[a] = rebinned_sd[:]
        else:
            rebinned = rebin(x1, y1[a], x2)

        output[a] = rebinned[:]
        it.iternext()

    if y1_sd is not None:
        return output, output_sd
    else:
        return output


def rebinND(y1, axes, old_bins, new_bins, y1_sd=None):
    """
    Rebin y1 along several axes, in a piecewise-constant fashion.

    Parameters
    ----------
    y1 : array_like
        The image to be rebinned
    axes : tuple of int
        The axes to be rebinned.
    old_bins : tuple of np.ndarray
        The old histogram bins along each axis in `axes`.
    new_bins : tuple of np.ndarray
        The new histogram bins along each axis in `axes`.
    y1_sd : array_like, optional
        Standard deviations for pixels in the image

    Returns
    -------
    output : np.ndarray
    --OR--
    (output, output_sd) : np.ndarray
        The rebinned image
    """
    num_axes = len(y1.shape)
    if np.max(axes) > num_axes - 1 or np.min(axes) < 0:
        raise ValueError("One of the axes is not in the original array")

    if (len(old_bins) != len(new_bins) or len(old_bins) != len(axes)):
        raise ValueError("The number of bins must be the same as the number"
                         "of axes you wish to rebin")

    output = np.copy(y1)
    for i, axis in enumerate(axes):
        output = rebin_along_axis(output,
                                  old_bins[i],
                                  new_bins[i],
                                  axis,
                                  y1_sd=y1_sd)

    return output


def rebin(x1, y1, x2, y1_sd=None):
    """
    Rebin histogram values y1 from old bin edges x1 to new edges x2.
    (Piecewise_constant)

    Parameters
    ----------
     x1 : np.ndarray
         M + 1 array of old bin edges.
     y1 : np.ndarray
        M + 1 array of old histogram values. This is the total number in
        each bin, not an average.
     x2 : np.ndarray
        N + 1 array of new bin edges.
     y1_sd : np.ndarray, optional
        Standard deviations of values in y1

    Returns
    -------
    y2 : np.ndarray
    --OR--
    (y2, y2_sd) : np.ndarray
        An array of rebinned histogram values.

    The rebinning algorithm assumes that the counts in each old bin are
    uniformly distributed in that bin.

    Bins in x2 that are entirely outside the range of x1 are assigned 0.
    """

    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)

    y1_sd_temp = y1_sd
    if y1_sd is None:
        y1_sd_temp = np.zeros_like(y1)

    # Divide y1 by bin widths.
    # This converts y-values from bin total to bin average over bin width.
    x1_bin_widths = np.ediff1d(x1)
    y1_ave = y1 / x1_bin_widths
    y1_ave_sd = y1_sd_temp / x1_bin_widths

    # Need to work with variances
    y1_ave_var = y1_ave_sd ** 2

    # allocating y2 vector
    n  = x2.size - 1
    y2 = np.zeros(n)
    y2_var = np.zeros_like(y2)

    i_place = np.searchsorted(x1, x2)

    # create a placeholder array for holding subset of fractional bin widths
    max_i_place_length = np.max(np.ediff1d(i_place))
    p_sub_dx = np.zeros(max_i_place_length + 1)

    # loop over all new bins
    for i in range(n):
        x2_lo, x2_hi = x2[i], x2[i + 1]

        i_lo, i_hi = i_place[i], i_place[i + 1]

        # new bin out of x1 range
        if i_hi == 0 or i_lo == x1.size:
            continue

        # new bin totally covers x1 range
        elif i_lo == 0 and i_hi == x1.size:
            sub_edges = x1
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave
            sub_y_ave_var = y1_ave_var

        # new bin overlaps lower x1 boundary
        elif i_lo == 0:
            sub_edges = np.hstack( [ x1[i_lo: i_hi], x2_hi ] )
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave[i_lo: i_hi]
            sub_y_ave_var = y1_ave_var[i_lo: i_hi]

        # new bin overlaps upper x1 boundary
        elif i_hi == x1.size:
            sub_edges = np.hstack( [ x2_lo, x1[i_lo: i_hi] ] )
            sub_dx    = np.ediff1d(sub_edges)
            sub_y_ave = y1_ave[i_lo - 1: i_hi]
            sub_y_ave_var = y1_ave_var[i_lo - 1: i_hi]

        # new bin is enclosed in x1 range
        else:
            # sub_edges = np.hstack( [ x2_lo, x1[i_lo: i_hi], x2_hi ] )
            # sub_dx    = np.ediff1d(sub_edges)

            # hstack and ediff1d are expensive operations. The bin widths have
            # already been calculated, so let's reuse them.
            p_sub_dx[0] = x1[i_lo] - x2_lo
            length = i_hi - i_lo
            p_sub_dx[1: length] = x1_bin_widths[i_lo: i_hi - 1]
            p_sub_dx[length] = x2_hi - x1[i_hi - 1]
            sub_dx = p_sub_dx[0: length + 1]

            sub_y_ave = y1_ave[i_lo - 1: i_hi]
            sub_y_ave_var = y1_ave_var[i_lo - 1: i_hi]

        y2[i] = np.sum(sub_dx * sub_y_ave)
        y2_var[i] = np.sum(sub_y_ave_var * (sub_dx ** 2))

    if y1_sd is None:
        return y2
    else:
        return y2, np.sqrt(y2_var)
