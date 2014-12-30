from __future__ import division
import numpy as np
from scipy.integrate import cumtrapz
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
	
def centroid(y, x = None):
	"""
	finds the centroid position of array a (assumed to be 1D), with unit spacing (by default)
	"""
	if not np.all(x):
		x = np.arange(1. * len(y))
	
	xsort = np.argsort(x)
	xvals = x[xsort]
	yvals = y[xsort]
	
	loc = cumtrapz(yvals, x = xvals)
	#get the cumulative distribution function
	loc -= loc[0]
	loc /= loc[-1]
	
	#the centroid is when the CDF=0.5.
	meanloc = 1 + np.interp(0.5, loc, np.arange(len(loc) * 1.))
	meanval = np.interp(meanloc, np.arange(len(xvals) * 1.), xvals)
	
	sdrhsloc = 1 + np.interp(0.84123, loc, np.arange(len(loc) * 1.))
	sdlhsloc = 1 + np.interp(0.15877, loc, np.arange(len(loc) * 1.))
	
	sdrhsval = np.interp(sdrhsloc, np.arange(len(xvals) * 1.), xvals)
	sdlhsval = np.interp(sdlhsloc, np.arange(len(xvals) * 1.), xvals)
	return meanval, 0.5 * (sdrhsval - sdlhsval)
	
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
	
def divergence(d1, d2, L12):
    return 2.35 / L12 * np.sqrt((np.pow(d1, 2) + np.pow(d2, 2))/L12)
	