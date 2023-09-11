import numpy as np


def get_scaling_in_overlap(x0, y0, dy0, x1, y1, dy1):
    r"""
    Obtain vertical scaling factor that splices the second dataset onto the
    first.

    Parameters
    ----------
    x0: np.ndarray
        abscissae for the first dataset
    y0: np.ndarray
        y values for the first dataset
    dy0: np.ndarray
        dy (standard deviation) values for the first dataset
    x1: np.ndarray
        abscissae values for the second dataset
    y1: np.ndarray
        y values for the second dataset
    dy1: np.ndarray
        dy (standard deviation) values for the second dataset

    Returns
    -------
    (scale, dscale, overlap_points): float, float, array-like
        `scale` and `dscale` are the scaling and uncertainty in scaling factor.
        They are `np.nan` if the abscissae ranges don't overlap.
        `overlap_points` indicates the points in the *first* dataset that are
        in the overlap region.
    """

    # the datasets should be sorted, but we may not want to sort the data
    # so make a temporary copy of the data
    use_dy = (dy1 is not None) and (dy0 is not None)

    sort_arr0 = np.argsort(x0)
    tx0 = x0[sort_arr0]
    ty0 = y0[sort_arr0]
    if dy0 is not None:
        tdy0 = dy0[sort_arr0]
    sort_arr1 = np.argsort(x1)
    tx1 = x1[sort_arr1]
    ty1 = y1[sort_arr1]
    if dy1 is not None:
        tdy1 = dy1[sort_arr1]

    # largest point number of x1 in overlap region
    num2 = tx1[tx1 < tx0[-1]].size

    if num2 == 0:
        return np.nan, np.nan, np.array([])

    # get scaling factor at each point of second dataset in the overlap region
    # get the intensity of wave1 at an overlap point
    newi = np.interp(tx1[:num2], tx0, ty0)
    w_scalefactor = newi / ty1[:num2]

    if use_dy:
        newdi = np.interp(tx1[:num2], tx0, tdy0)
        w_dscalefactor = np.sqrt(
            (newdi / ty1[:num2]) ** 2
            + ((newi * tdy1[:num2]) ** 2) / ty1[:num2] ** 4
        )

        w_dscalefactor = 1 / (w_dscalefactor**2)
    else:
        w_dscalefactor = 1.0

    num = np.sum(w_scalefactor * w_dscalefactor)
    den = np.sum(w_dscalefactor)

    normal = num / den
    dnormal = np.sqrt(1 / den)

    # work out the points in x0 which overlap with x1. If they overlap you may
    # be able to delete them.
    overlap_points = np.where(x0 > tx1[0], True, False)

    return normal, dnormal, overlap_points
