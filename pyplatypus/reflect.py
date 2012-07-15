from __future__ import division
import numpy as np
import numpy.matlib as npmat
import math

def reflectivity(coefs, qvals):
	
	if np.size(coefs, 0) != 4 * coefs[0] + 8:
		raise Exception('coefs the wrong size')
		
	nlayers = int(coefs[0])
	npnts = len(qvals)
	
	pj = np.zeros((npnts, nlayers + 2), dtype = 'complex')
	roughnesses = np.zeros(nlayers + 1, 'float')
#	rj = np.array((len(qvals), nlayers + 1), 'complex')
	
	SLDfronting = np.complex(coefs[2] * 1.e-6 + coefs[3] * j)
	SLDbacking = np.complex(coefs[4] * 1.e-6 + coefs[5] * j)
	
	for layer in xrange(coefs[0]):
		pj[:, layer + 1] = -4. * math.pi * (coefs[4 * layer + 9] * 1.e-6 + coefs[4*layer + 10] * j - SLDfronting)
		roughnesses[layer] = np.fabs(coefs[4 * layer + 11])
		
	roughnesses[-1] = coefs[7]
	
	pj[:, numlayers + 1] = -4 * math.pi * (SLDbacking - SLDfronting);

	qq2 = np.power(qvals, 2.) / 4.
	pj[:,:] += qq2[:, np.newaxis]
	
	pj = np.sqrt(pj)

	rj = (pj[:, :-1] - pj[:, 1:]) / (pj[:, :-1] + pj[:, 1:])

	rj *= np.exp(pj[:, :-1] * pj[:, 1:] * -2 * roughnesses)
	
	MRtotal = np.zeros((npnts, 2, 2), dtype = 'complex')
	MRtotal[:] = npmat.eye(2)
	