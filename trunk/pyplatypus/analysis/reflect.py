from __future__ import division
import numpy as np
import math
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
    
    b = np.arange(1000.)
    b /= 2000.
    b += 0.001
    	
    def loop():
        abeles(b, a)
    	
    t = timeit.Timer(stmt = loop)
    print t.timeit(number = 1000)
