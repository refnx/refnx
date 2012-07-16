from __future__ import division
import numpy as np
import numpy.matlib as npmat
import math

def reflectivity(coefs, qvals):
	
	if np.size(coefs, 0) != 4 * coefs[0] + 8:
		raise Exception('coefs the wrong size')
		
	nlayers = int(coefs[0])
	npnts = len(qvals)
	
	pj = np.zeros((npnts, nlayers + 2), dtype = 'complex128')
	roughnesses = np.zeros(nlayers + 1)
	
	SLDfronting = np.complex(coefs[2] * 1.e-6, coefs[3])
	SLDbacking = np.complex(coefs[4] * 1.e-6, coefs[5])
	
	for layer in xrange(nlayers):
		pj[:, layer + 1] = -4. * math.pi * (np.complex(coefs[4 * layer + 9] * 1.e-6, coefs[4 * layer + 10]) - SLDfronting)
		roughnesses[layer] = np.fabs(coefs[4 * layer + 11])
		
	roughnesses[-1] = coefs[7]
	pj[:, nlayers + 1] = -4 * math.pi * (SLDbacking - SLDfronting);
	
	qq2 = np.power(qvals, 2.) / 4.
	
	pj[:,:] += qq2[:, np.newaxis]	
	pj = np.sqrt(pj)
	
	#work out the fresnel reflection for each layer
	rj = (pj[:, :-1] - pj[:, 1:]) / (pj[:, :-1] + pj[:, 1:])
	rj *= np.exp(pj[:, :-1] * pj[:, 1:] * -2. * roughnesses * roughnesses)	
		
	MRtotal = np.zeros((npnts, 2, 2), dtype = 'complex')
	MRtotal[:] = npmat.eye(2)

	MI = np.zeros((npnts, 2, 2), dtype = 'complex')
	
	for layer in xrange(nlayers + 1):
		if not layer:
			beta = np.complex(1., 0.)
		else:
			beta = np.exp(pj[:, layer] * np.complex(0, math.fabs(coefs[4 * layer + 4])))
		
		MI[:, 0, 0] = beta
		MI[:, 0, 1] = rj[:, layer] * beta;
		MI[:, 1, 1] = 1. / beta;
		MI[:, 1, 0] = rj[:, layer] / beta;
				
#		totally weird way of matrix multiplication
#		http://jameshensman.wordpress.com/2010/06/14/multiple-matrix-multiplication-in-numpy/
		MRtotal = np.sum(np.transpose(MRtotal, (0, 2, 1)).reshape(npnts, 2, 2, 1) * MI.reshape(npnts, 2, 1, 2), -3)
		
	#now work out the reflectivity
	reflectivity = (MRtotal[:, 1, 0] * np.conj(MRtotal[:, 1, 0])) /  (MRtotal[:, 0, 0] * np.conj(MRtotal[:, 0, 0]))

	return np.asarray(reflectivity, dtype='float')
	


