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
        # Using a non-tuple sequence for multidimensional indexing is
        # deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the
        # future this will be interpreted as an array index,
        # `arr[np.array(seq)]`, which will result either in an error or a
        # different result.
        b = tuple(a)
        if y1_sd is not None:
            rebinned, rebinned_sd = rebin(x1,
                                          y1[b],
                                          x2,
                                          y1_sd=y1_sd[b])

            output_sd[b] = rebinned_sd[:]
        else:
            rebinned = rebin(x1, y1[b], x2)

        output[b] = rebinned[:]
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
    y2 or (y2, y2_sd) : np.ndarray
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
    x1_bin_widths = np.diff(x1)
    y1_ave = y1 / x1_bin_widths
    y1_ave_sd = y1_sd_temp / x1_bin_widths

    # Need to work with variances
    y1_ave_var = y1_ave_sd ** 2
    y1_var_temp = y1_sd_temp ** 2

    # allocating y2 vector
    n = x2.size - 1
    y2 = np.zeros(n)
    y2_var = np.zeros_like(y2)

    i_place = np.searchsorted(x1, x2)

    # find out where x2 intersects with x1, this will determine which x2 bins
    # we need to consider
    start_pos = 0
    end_pos = n

    start_pos_test = np.where(i_place == 0)[0]
    if start_pos_test.size > 0:
        start_pos = start_pos_test[-1]

    end_pos_test = np.where(i_place == x1.size)[0]
    if end_pos_test.size > 0:
        end_pos = end_pos_test[0]

    # the first bin totally covers x1 range
    if (start_pos == end_pos - 1 and
            i_place[start_pos] == 0 and
            i_place[start_pos + 1] == x1.size):
        sub_edges = x1
        sub_dx = np.diff(sub_edges)
        sub_y_ave = y1_ave
        sub_y_ave_var = y1_ave_var

        y2[start_pos] = np.sum(sub_dx * sub_y_ave)
        y2_var[start_pos] = np.sum(sub_y_ave_var * (sub_dx ** 2))

        start_pos = end_pos

    # the first bin overlaps lower x1 boundary
    if i_place[start_pos] == 0 and start_pos < end_pos:
        x2_lo, x2_hi = x2[start_pos], x2[start_pos + 1]
        i_lo, i_hi = i_place[start_pos], i_place[start_pos + 1]

        sub_edges = np.hstack([x1[i_lo:i_hi], x2_hi])
        sub_dx = np.diff(sub_edges)
        sub_y_ave = y1_ave[i_lo: i_hi]
        sub_y_ave_var = y1_ave_var[i_lo:i_hi]

        y2[start_pos] = np.sum(sub_dx * sub_y_ave)
        y2_var[start_pos] = np.sum(sub_y_ave_var * (sub_dx ** 2))

        start_pos += 1

    # the last bin overlaps upper x1 boundary
    if (i_place[end_pos] == x1.size and start_pos < end_pos):
        x2_lo, x2_hi = x2[end_pos - 1], x2[end_pos]
        i_lo, i_hi = i_place[end_pos - 1], i_place[end_pos]

        sub_edges = np.hstack([x2_lo, x1[i_lo:i_hi]])
        sub_dx = np.diff(sub_edges)
        sub_y_ave = y1_ave[i_lo - 1:i_hi]
        sub_y_ave_var = y1_ave_var[i_lo - 1:i_hi]

        y2[end_pos - 1] = np.sum(sub_dx * sub_y_ave)
        y2_var[end_pos - 1] = np.sum(sub_y_ave_var * (sub_dx ** 2))

        end_pos -= 1

    if start_pos < end_pos:
        # deal with whole parts of bin that are spanned
        cum_sum = np.cumsum(y1)
        cum_sum_var = np.cumsum(y1_var_temp)
        running_sum = (cum_sum[i_place[start_pos + 1:end_pos + 1] - 2] -
                       cum_sum[i_place[start_pos:end_pos] - 1])
        running_sum_var = (
            cum_sum_var[i_place[start_pos + 1:end_pos + 1] - 2] -
            cum_sum_var[i_place[start_pos:end_pos] - 1])

        y2[start_pos:end_pos] += running_sum
        y2_var[start_pos:end_pos] += running_sum_var

        # deal with fractional start of bin
        p_sub_dx = x1[i_place[start_pos:end_pos]] - x2[start_pos:end_pos]
        sub_y_ave = y1_ave[i_place[start_pos:end_pos] - 1]
        sub_y_ave_var = y1_ave_var[i_place[start_pos:end_pos] - 1]

        y2[start_pos:end_pos] += p_sub_dx * sub_y_ave
        y2_var[start_pos:end_pos] += sub_y_ave_var * (p_sub_dx) ** 2

        # deal with fractional end of bin
        p_sub_dx = (x2[start_pos + 1:end_pos + 1] -
                    x1[i_place[start_pos + 1:end_pos + 1] - 1])
        sub_y_ave = y1_ave[i_place[start_pos + 1:end_pos + 1] - 1]
        sub_y_ave_var = y1_ave_var[i_place[start_pos + 1:end_pos + 1] - 1]

        y2[start_pos:end_pos] += p_sub_dx * sub_y_ave
        y2_var[start_pos:end_pos] += sub_y_ave_var * (p_sub_dx) ** 2

    if y1_sd is None:
        return y2
    else:
        return y2, np.sqrt(y2_var)
