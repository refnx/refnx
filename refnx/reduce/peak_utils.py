from __future__ import division
import numpy as np
from scipy.integrate import simps
from scipy.optimize import curve_fit
import random
import hashlib
#from scipy.interpolate import interp1d


#def searchinterp(a, val):
#	"""
#	finds an interpolated x position of val array in a array
#	returns an interpolated position.  If the value is before the first value in the array,
#	 or if the value is after the last point in the array, location will be NaN
#	"""	   
#	f = interp1d(np.arange(np.size(a, 0)) * 1., a, bounds_error = False)
#	
#	return f(val)

def centroid(y, x=None, dx=1.):
    '''Computes the centroid for the specified data.

    Parameters
    ----------
    y : array_like
        Array whose centroid is to be calculated.
    x : array_like, optional
        The points at which y is sampled.
    Returns
    -------
    (centroid, sd)
        Centroid and standard deviation of the data.
    '''
    yt = np.array(y)

    if x is None:
        x = np.arange(yt.size, dtype='float') * dx

    normaliser = simps(yt, x)
    centroid = simps(x * yt, x) / normaliser
    var = simps((x - centroid)**2 * yt, x) / normaliser
    return centroid, np.sqrt(var)
	
def gauss_fit(p0, x, y, sigma = None):
	popt, pcov = curve_fit(gauss, x, y, p0 = p0, sigma = sigma)
	return popt
	
def gauss(x, bkg, peak, mean, sd):
	return bkg + peak * np.exp(-0.5 * ((mean - x) / sd)**2) 
	
def peak_finder(y, x = None):
	maxval = np.amax(y)
	if not x:
		x = np.arange(1. * len(y))
	
	expected_centre, expected_SD = centroid(y, x = x)
	
	p0 = np.array([2., maxval, expected_centre, expected_SD])
	popt = gauss_fit(p0, x, y)
	return np.array([expected_centre, expected_SD]), popt[2:4]

def peakfinder_test():
	random.seed()
	x = np.random.uniform(-10, 10, 300)
	y = gauss(x, 0.2, 10, 2, 1.25) + random.gauss(0, 0.5)
	peak_finder(y, x = x)
