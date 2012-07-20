from __future__ import division
import numpy as np
import math
try:
    import _creflect.abeles as refcalc
except ImportError:
    import _reflect as refcalc


def abeles(coefs, qvals, dqvals = None):
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
	
	return refcalc.abeles(np.size(qvals, 0), coefs, qvals)
	

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
        abeles(a, b)
    	
    t = timeit.Timer(stmt = loop)
    print t.timeit(number = 1000)
