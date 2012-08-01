from __future__ import division
import numpy as np
import math
import pyplatypus.analysis.energyfunctions as energyfunctions
from scipy.stats import norm

try:
	import _creflect as refcalc
except ImportError:
	import _reflect as refcalc


def abeles(qvals, coefs, *args, **kwds):
	"""

	Abeles matrix formalism for calculating reflectivity from a stratified medium.

	coefs - :
	coefs[0] = number of layers, N
	coefs[1] = scale factor
	coefs[2] = SLD of fronting (/1e-6 Angstrom**-2)
	coefs[3] = iSLD of fronting (/Angstrom**-2)
	coefs[4] = SLD of backing
	coefs[5] = iSLD of backing
	coefs[6] = background
	coefs[7] = roughness between backing and layer N

	coefs[4 * (N - 1) + 8] = thickness of layer N in Angstrom (layer 1 is closest to fronting)
	coefs[4 * (N - 1) + 9] = SLD of layer N
	coefs[4 * (N - 1) + 10] = iSLD of layer N
	coefs[4 * (N - 1) + 11] = roughness between layer N and N-1.

	qvals - the qvalues required for the calculation. Q=4*Pi/lambda * sin(omega). Units = Angstrom**-1

	"""
	
	if 'dqvals' in kwds and kwds['dqvals'] is not None:
		dqvals = kwds['dqvals']
		weights = np.array([0.0404840047653159,
			0.0921214998377285,
			0.1388735102197872,
			0.1781459807619457,
			0.2078160475368885,
			0.2262831802628972,
			0.2325515532308739,
			0.2262831802628972,
			0.2078160475368885,
			0.1781459807619457,
			0.1388735102197872,
			0.0921214998377285,
			0.0404840047653159])

		abscissa = np.array([-0.9841830547185881,
			-0.9175983992229779,
			-0.8015780907333099,
			-0.6423493394403402,
			-0.4484927510364469,
			-0.2304583159551348,
			0.,
			0.2304583159551348,
			0.4484927510364469,
			0.6423493394403402,
			0.8015780907333099,
			0.9175983992229779,
			0.9841830547185881])

		gaussvals  = np.array([0.001057642102668805,
			0.002297100003792314,
			0.007793859679303332,
			0.0318667809686739,
			0.1163728244269813,
			0.288158781825899,
			0.3989422804014327,
			0.288158781825899,
			0.1163728244269813,
			0.0318667809686739,
			0.007793859679303332,
			0.002297100003792314,
			0.001057642102668805])

		#integration between -3.5 and 3 sigma
		INTLIMIT = 3.5
		FWHM = 2 * math.sqrt(2 * math.log(2.0))
				
		va = qvals.flatten() - INTLIMIT * dqvals /FWHM
		vb = qvals.flatten() + INTLIMIT * dqvals /FWHM

		va = va[:, np.newaxis]
		vb = vb[:, np.newaxis]
			  
		qvals_for_res = (np.atleast_2d(abscissa) * (vb - va) + vb + va) / 2.        

		smeared_rvals = refcalc.abeles(np.size(qvals_for_res.flatten(), 0), qvals_for_res.flatten(), coefs)
		smeared_rvals = np.reshape(smeared_rvals, (qvals.size, abscissa.size))

		smeared_rvals *= np.atleast_2d(gaussvals * weights)

		return np.sum(smeared_rvals, 1) * INTLIMIT
	else:
		return refcalc.abeles(np.size(qvals.flatten(), 0), qvals.flatten(), coefs)

def sld_profile(coefs, z):
		
	nlayers = int(coefs[0])
	dist = 0
	summ = coefs[2]
	
	for ii in xrange(nlayers + 1):
		if ii == 0:
			if nlayers:
				deltarho = -coefs[2] + coefs[9]
				thick = 0
				sigma = math.fabs(coefs[11])
			else: 
				sigma = math.fabs(coefs[7])
				deltarho = -coefs[2] + coefs[4]
		elif ii == nlayers:
			SLD1 = coefs[4 * ii + 5]
			deltarho = -SLD1 + coefs[4]
			thick = math.fabs(coefs[4 * ii + 4])
			sigma = math.fabs(coefs[7])
		else:
			SLD1 = coefs[4 * ii + 5]
			SLD2 = coefs[4 * (ii + 1) + 5]
			deltarho = -SLD1 + SLD2
			thick = math.fabs(coefs[4 * ii + 4])
			sigma = math.fabs(coefs[4 * (ii + 1) + 7])

		dist += thick
	
		#if sigma=0 then the computer goes haywire (division by zero), so say it's vanishingly small
		if sigma == 0:
			sigma += 1e-3
	
		summ += deltarho * (norm.cdf((z - dist)/sigma))		
		
	return summ


class ReflectivityFitObject(energyfunctions.FitObject):
	
	'''
		A sub class of pyplatypus.analysis.energyfunctions.FitObject suited for fitting reflectometry data.
		The main difference is that the energy of the cost function is log10 scaled: (log10(calc) - log10(model))**2
		This fit object does _not_ use the error bars on each of the data points.
		
		
	'''
	
	def __init__(self, xdata, ydata, edata, parameters, *args, **kwds):
		super(ReflectivityFitObject, self).__init__(xdata, ydata, edata, None, parameters, *args, **kwds)

	def energy(self, params = None):
		"""
			The default cost function for the reflectivity object is chi2.
			params - np.ndarray containing the parameters that are being fitted, i.e. this array is np.size(self.fitted_parameters) long.
			Returns chi2.
		
		"""
		
		test_parameters = np.copy(self.parameters)
	
		if params is not None:
			test_parameters[self.fitted_parameters] = params
		
		model = abeles(self.xdata, test_parameters, *self.args, **self.kwds)			
			
		return  np.sum(np.power(np.log10(self.ydata) - np.log10(model), 2))

	
	
if __name__ == '__main__':
	import timeit
	a = np.zeros((12))
	a[0] = 1.
	a[1] = 1.
	a[4] = 2.07
	a[7] = 3
	a[8] = 100
	a[9] = 3.47
	a[11] = 2

	b = np.arange(10000.)
	b /= 20000.
	b += 0.0005
 
	def loop():
		abeles(b, a)

	t = timeit.Timer(stmt = loop)
	print t.timeit(number = 10000)
