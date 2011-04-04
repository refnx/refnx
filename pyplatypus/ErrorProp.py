from __future__ import division
import numpy as np

#http://en.wikipedia.org/wiki/Propagation_of_uncertainty

def EPadd(a, da, b, db, covar = 0):
	"""
	C = A + B
	"""
	return (a + b), np.sqrt(da**2 + db**2 + 2 * covar)
	
def EPsub(a, da, b, db, covar = 0):
	"""
	C = A - B
	"""
	return a - b, np.sqrt(da**2 + db**2 + 2 * covar)
	
def EPmul(a, da, b, db, covar = 0):
	"""
	C = A * B
	"""
	return a * b, np.fabs((a * b) * np.sqrt((da / a)**2 + (db / b)**2 + 2 * da * db / (a * b) * covar))
	
def EPmulk(a, da, k):
	"""
	C = A * k
	"""
	return a * k, np.fabs(da * k)
	
def EPdiv(a, da, b, db, covar = 0):
	"""
	C = A / B
	"""
	return a / b, np.fabs((a / b) * np.sqrt((da / a)**2 + (db / b)**2 - 2 * da * db / (a * b) * covar))

def EPpow(a, da, k, n = 1):
	"""
	C = n * (A**k)
	"""
	return n * np.power(a, k), np.fabs(n * np.power(a, k) * j * da / a)
	
def EPpowk(a, da, k, n = 1):
	"""
	C = pow(k, A * n)
	"""
	return np.power(k, a * n), np.fabs(np.power(k, a * n) * n * da * np.log(k))
	
def EPlog(a, da, k = 1, n = 1):
	"""
	C = n * log(k * A )
	"""
	return n * np.log(k * a), np.fabs(n * da /a)

def EPlog10(a, da, k = 1, n = 1):
	"""
	C = n * log10(k * A)
	"""
	return n * np.log10(k * a), np.fabs(n * da / (a * np.log(10)))
	
def EPexp(a, da, k = 1, n = 1):
	"""
	C = n * exp(k * A)
	"""
	return n * np.exp(k * a), np.fabs(k * da * n * np.exp(k * a))
	
def EPsin(a, da):
	"""
	C = sin (A)
	"""
	return np.sin(a), np.fabs(cos(a) * da)

def EPcos(a, da):
	"""
	C = cos (A)
	"""
	return np.cos(a), np.fabs(-np.sin(a) * da)
