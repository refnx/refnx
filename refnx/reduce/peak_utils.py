import numpy as np
from scipy.integrate import simps, cumtrapz
from scipy.optimize import curve_fit


def centroid(y, x=None, dx=1.0):
    """Computes the centroid for the specified data.

    Parameters
    ----------
    y : array_like
        Array whose centroid is to be calculated.
    x : array_like, optional
        The points at which y is sampled.
    Returns
    -------
    (centroid, sd)
        Centroid and standard deviation of the data. If there are no
        intensities at all in `y`, then `centroid` and `sd` are `np.nan`.

    This is not really all that good of an algorithm, unless the peak is much
    higher than the background
    """
    yt = np.asfarray(y)

    if x is None:
        x = np.arange(yt.size, dtype="float") * dx

    normaliser = simps(yt, x)

    if normaliser == 0:
        return np.nan, np.nan

    centroid = simps(x * yt, x)

    centroid /= normaliser

    var = simps((x - centroid) ** 2 * yt, x) / normaliser

    return centroid, np.sqrt(var)


def median(y, x=None, dx=1.0):
    r"""
    Computes the median of the specified data.

    Parameters
    ----------
    y : array_like
        Array whose median is to be calculated.
    x : array_like, optional
        The points at which y is sampled.
    Returns
    -------
    (median, sd) : float, float
        Centroid and standard deviation of the data.

    """
    yt = np.asfarray(y)

    if x is None:
        x = np.arange(yt.size, dtype="float") * dx

    c = cumtrapz(yt, x=x, initial=0)
    c0 = c[0]
    cl = c[-1]
    c -= c0
    c /= cl

    median = np.interp(0.5, c, x)
    mean, sd = centroid(y, x=x)
    return median, sd


def gauss_fit(p0, x, y, sigma=None):
    popt, pcov = curve_fit(gauss, x, y, p0=p0, sigma=sigma)
    return popt


def gauss(x, bkg, peak, mean, sd):
    """Computes the gaussian function.

    Parameters
    ----------
    x : array_like
        The points at which the distribution is sampled.
    bkg : float
        constant background value.
    peak : float
        peak multiplier.
    mean : float
        mean of gaussian distribution.
    sd : float
        standard deviation of distribution.

    Returns
    -------
    gval : float
        evaluated gaussian distribution value at each of the sampling points.
    """

    return bkg + peak * np.exp(-0.5 * ((mean - x) / sd) ** 2)


def peak_finder(y, x=None, sigma=None):
    """Finds a peak in the specified data.

    Parameters
    ----------
    y : array_like
        Array in which a peak is to be found.
    x : array_like, optional
        The points at which y is sampled.
    Returns
    -------
    (centroid, sd), (gfit_mean, gfit_sd) : (float, float), (float, float)
    centroid and sd are the centroid and standard deviation of the data.
    gfit_mean and gfit_sd are the mean and standard deviation obtained by
    fitting the data to a Gaussian function.
    """
    maxval = np.amax(y)
    if x is None:
        x = np.arange(1.0 * len(y))

    expected_centre, expected_SD = centroid(y, x=x)

    try:
        p0 = np.array([2.0, maxval, expected_centre, expected_SD])
        popt = gauss_fit(p0, x, y)
    except RuntimeError:
        # if we can't find a centre return the centroid
        popt = p0
        popt[2:4] = expected_centre, expected_centre

    return np.array([expected_centre, expected_SD]), popt[2:4]
