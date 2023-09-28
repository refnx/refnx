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
        raise ValueError(
            "The original number of xbins does not match the axis size"
        )

    odtype = np.dtype("float")
    if y1.dtype is np.dtype("O"):
        odtype = np.dtype("O")

    output = np.empty(new_shape, dtype=odtype)
    output_sd = np.copy(output)

    it = np.nditer(y1, flags=["multi_index", "refs_ok"])
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
            rebinned, rebinned_sd = rebin(x1, y1[b], x2, y1_sd=y1_sd[b])

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

    if len(old_bins) != len(new_bins) or len(old_bins) != len(axes):
        raise ValueError(
            "The number of bins must be the same as the number"
            "of axes you wish to rebin"
        )

    output = np.copy(y1)
    for i, axis in enumerate(axes):
        output = rebin_along_axis(
            output, old_bins[i], new_bins[i], axis, y1_sd=y1_sd
        )

    return output


def rebin(x1, y1, x2, y1_sd=None):
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)

    y1_sd_temp = np.asarray(y1_sd)
    if y1_sd is None:
        y1_sd_temp = np.zeros_like(y1)

    # Need to work with variances
    y1_var_temp = y1_sd_temp**2

    # the fractional bin locations of the new bins in the old bins
    i_place = np.interp(x2, x1, np.arange(len(x1)))

    cum_sum = np.r_[[0], np.cumsum(y1)]
    cum_sum_var = np.r_[[0], np.cumsum(y1_var_temp)]

    start = np.interp(i_place[:-1], np.arange(len(cum_sum)), cum_sum)
    finish = np.interp(i_place[1:], np.arange(len(cum_sum)), cum_sum)

    y2 = finish - start

    if y1_sd is None:
        return y2

    # calculate variances for bins where lower and upper bin edges span
    # greater than or equal to one original bin.
    # This is the contribution from the 'intact' bins (not including the
    # fractional start and end parts.
    whole_bins = np.floor(i_place[1:]) - np.ceil(i_place[:-1]) >= 1.0

    start = cum_sum_var[np.ceil(i_place[:-1]).astype(int)]
    finish = cum_sum_var[np.floor(i_place[1:]).astype(int)]

    # start = np.interp(np.ceil(i_place[:-1]),
    #                   np.arange(len(cum_sum_var)),
    #                   cum_sum_var)
    #
    # finish = np.interp(np.floor(i_place[1:]),
    #                    np.arange(len(cum_sum_var)),
    #                    cum_sum_var)

    y2_var = np.where(whole_bins, finish - start, 0.0)

    bin_loc = np.clip(np.floor(i_place).astype(int), 0, len(y1_sd_temp) - 1)

    # fractional contribution for bins where the new bin edges are in the same
    # original bin.
    same_cell = np.floor(i_place[1:]) == np.floor(i_place[:-1])
    frac = i_place[1:] - i_place[:-1]
    contrib = (frac * y1_sd_temp[bin_loc[:-1]]) ** 2
    y2_var += np.where(same_cell, contrib, 0.0)

    # fractional contribution for bins where the left and right bin edges are
    # in different original bins.
    different_cell = np.floor(i_place[1:]) > np.floor(i_place[:-1])
    frac_left = np.ceil(i_place[:-1]) - i_place[:-1]
    contrib = (frac_left * y1_sd_temp[bin_loc[:-1]]) ** 2

    frac_right = i_place[1:] - np.floor(i_place[1:])
    contrib += (frac_right * y1_sd_temp[bin_loc[1:]]) ** 2

    y2_var += np.where(different_cell, contrib, 0.0)

    return y2, np.sqrt(y2_var)
