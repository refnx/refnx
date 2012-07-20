from __future__ import division
import numpy as np
import math


def abeles(lenqvals, qvals, coefs):
	"""
	
	   Abeles matrix formalism for calculating reflectivity from a stratified medium.
	   
	   lenqvals - the length of reflectivity values expected, should be len(qvals).  This 
	           is only required because the cReflect version is SWIGged and needs to supply as it's
	           first argument the size of the array to be returned. This value does nothing in the
	           pure python implementation. However, if you get it wrong in the cReflect version then
	           you will have memory leaks. 
	   
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
	
	if np.size(coefs, 0) != 4 * coefs[0] + 8:
		raise Exception('coefs the wrong size')
		
	nlayers = int(coefs[0])
	npnts = len(qvals)
	
	pj = np.zeros((npnts, nlayers + 2), dtype = 'complex128')
	roughnesses = np.zeros(nlayers + 1)
	
	SLDfronting = np.complex(coefs[2] * 1.e-6, coefs[3])
	SLDbacking = np.complex(coefs[4] * 1.e-6, coefs[5])

	qq2 = np.power(qvals, 2.) / 4.
	
	for layer in xrange(nlayers):
		pj[:, layer + 1] = - 4. * math.pi * (np.complex(coefs[4 * layer + 9] * 1.e-6, coefs[4 * layer + 10]) - SLDfronting)
		roughnesses[layer] = np.fabs(coefs[4 * layer + 11])
		
	roughnesses[-1] = coefs[7]
	pj[:, nlayers + 1] = -4 * math.pi * (SLDbacking - SLDfronting);
		
	pj[:,:] += qq2[:, np.newaxis]	
	pj = np.sqrt(pj)
	
	#work out the fresnel reflection for each layer
	rj = np.exp(pj[:, :-1] * pj[:, 1:] * -2. * roughnesses * roughnesses) * (pj[:, :-1] - pj[:, 1:]) / (pj[:, :-1] + pj[:, 1:])
		
	MRtotal = np.zeros((npnts, 2, 2), dtype = 'complex')
	MI = np.zeros_like(MRtotal)

	MRtotal[:, 0, 0] = 1.
	MRtotal[:, 1, 1] = 1.
	
	for layer in xrange(nlayers + 1):
		if not layer:
			beta = np.complex(1., 0.)
		else:
			beta = np.exp(pj[:, layer] * np.complex(0, math.fabs(coefs[4 * layer + 4])))
		
		MI[:, 0, 0] = beta
		MI[:, 0, 1] = rj[:, layer] * beta;
		MI[:, 1, 1] = 1. / beta;
		MI[:, 1, 0] = rj[:, layer] * MI[:, 1, 1];
				
#		totally weird way of matrix multiplication
#		http://jameshensman.wordpress.com/2010/06/14/multiple-matrix-multiplication-in-numpy/
		MRtotal = np.sum(np.transpose(MRtotal, (0, 2, 1)).reshape(npnts, 2, 2, 1) * MI.reshape(npnts, 2, 1, 2), -3)

			
	#now work out the reflectivity
	reflectivity = (MRtotal[:, 1, 0] * np.conj(MRtotal[:, 1, 0])) /  (MRtotal[:, 0, 0] * np.conj(MRtotal[:, 0, 0]))

	return reflectivity.real
	

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
        abeles(len(b), b, a)
    	
    t = timeit.Timer(stmt = loop)
    print t.timeit(number = 1000)
