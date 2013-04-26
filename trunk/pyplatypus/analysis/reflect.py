from __future__ import division
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.integrate as spi
import math
import pyplatypus.analysis.fitting as fitting
import pyplatypus.util.ErrorProp as EP
import warnings

#from scipy.stats import norm

try:
    import pyplatypus.analysis._creflect as refcalc
except ImportError:
    import pyplatypus.analysis._reflect as refcalc

FWHM = 2 * math.sqrt(2 * math.log(2.0))
INTLIMIT = 3.5

def gauss_legendre(n):
    '''
        a function return gaussian quadrature weights
    '''
    k = np.arange(1.0, n)       
    a_band = np.zeros((2, n)) 
    a_band[1, 0 : n-1] = k / np.sqrt(4 * k * k - 1) 
    x, V = sp.linalg.eig_banded(a_band, lower = True) 
    w = 2 * np.real(np.power(V[0, :], 2)) 
    return x, w

def _smearkernel(x, coefs, q, dq):
    prefactor = 1 / math.sqrt(2 * math.pi)
    gauss = prefactor * np.exp(-0.5 * x * x)
    localq = q + x * dq/FWHM
    return refcalc.abeles(np.size(localq), localq, coefs) * gauss
    
def abeles(q, coefs, *args, **kwds):
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
    
    kwds['dqvals'] - an array containing the FWHM of the Gaussian approximated resolution kernel. Has the same size as qvals.
    kwds['quad_order'] - the order of the Gaussian quadrature polynomial for doing the resolution smearing. default = 17. Don't choose less than 13. If quad_order == 'ultimate' then adaptive quadrature is used. Adaptive quadrature will always work, but takes a _long_ time (2 or 3 orders of magnitude longer). Fixed quadrature will always take a lot less time. BUT it won't necessarily work across all samples. For example, 13 points may be fine for a thin layer, but will be atrocious at describing a multilayer with bragg peaks.

    """
    qvals = q.flatten()
    quad_order = 17
    
    if np.size(coefs, 0) != 4 * int(coefs[0]) + 8:
        raise ValueError("The size of the parameter array passed to abeles should be 4 * coefs[0] + 8")
    
    if 'quad_order' in kwds:
        quad_order = kwds['quad_order']
    
    if 'dqvals' in kwds and kwds['dqvals'] is not None:
        dqvals = kwds['dqvals']
        
        if quad_order == 'ultimate':
            #adaptive gaussian quadrature.
            smeared_rvals = np.zeros(qvals.size)
            warnings.simplefilter('ignore', Warning)
            for idx, val in enumerate(qvals):                
                smeared_rvals[idx], err = scipy.integrate.quadrature(_smearkernel, -INTLIMIT, INTLIMIT, tol=2 * np.finfo(np.float64).eps, rtol = 2 * np.finfo(np.float64).eps, args=(coefs, qvals[idx], dqvals[idx]))
            
            warnings.resetwarnings()
            return smeared_rvals
        else:
            #just do gaussian quadrature of fixed order
            #get the gauss-legendre weights and abscissa
            abscissa, weights = gauss_legendre(quad_order)
            #get the normal distribution at that point
            prefactor = 1 / math.sqrt(2 * math.pi)
            gauss = lambda x : np.exp(-0.5 * x * x)
            gaussvals = prefactor * gauss(abscissa * INTLIMIT)
            
            #integration between -3.5 and 3 sigma
            va = qvals - INTLIMIT * dqvals /FWHM
            vb = qvals + INTLIMIT * dqvals /FWHM

            va = va[:, np.newaxis]
            vb = vb[:, np.newaxis]
              
            qvals_for_res = (np.atleast_2d(abscissa) * (vb - va) + vb + va) / 2.        
            smeared_rvals = refcalc.abeles(np.size(qvals_for_res.flatten(), 0), qvals_for_res.flatten(), coefs)
            smeared_rvals = np.reshape(smeared_rvals, (qvals.size, abscissa.size))

            smeared_rvals *= np.atleast_2d(gaussvals * weights)

            return np.sum(smeared_rvals, 1) * INTLIMIT
    else:
        return refcalc.abeles(np.size(qvals, 0), qvals, coefs)

def isProperAbelesInput(coefs):
	'''
		a test to see if the coefs array is suitable input for the abeles function
	'''
	if np.size(coefs, 0) != 4 * int(coefs[0]) + 8:
		return False
	return True
	

def sld_profile(coefs, z):
        
    nlayers = int(coefs[0])
    summ = np.zeros_like(z)
    summ += coefs[2]
    thick = 0
    
    #note that you can do this in a single loop, which would save a lot of time,
    #using the scipy.norm.cdf function. However, if you use py2app it included 
    #the entirety of that package, bloating the file by 70Mb.
    #Using a nested loop reduces file size, and does not cause a significant performance
    #penalty
    
    for idx, zed in enumerate(z):
        dist = 0
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
        
            #summ += deltarho * (norm.cdf((zed - dist)/sigma))  
            summ[idx] += deltarho * (0.5 + 0.5 * math.erf((zed - dist)/(sigma * math.sqrt(2.))))     
        
    return summ


class ReflectivityFitObject(fitting.FitObject):
    
    '''
        A sub class of pyplatypus.analysis.energyfunctions.FitObject suited for fitting reflectometry data. 

        If you wish to fit analytic profiles you should subclass this fitobject, overriding the model() method
        of the FitObject super class.  If you do this you should also override the sld_profile method of ReflectivityFitObject.
    '''
    
    def __init__(self, xdata, ydata, edata, parameters, args = (), **kwds):
        '''
            Initialises the ReflectivityFitObject.
            See the constructor of the FitObject for more details. And possible values for the keyword args for the superclass.
        '''
        super(ReflectivityFitObject, self).__init__(xdata, ydata, edata, abeles, parameters, args = args, **kwds)
        
    def sld_profile(self, test_parameters, args = (), **kwds):
        """
            returns the SLD profile corresponding to the model parameters.
            The model parameters are either taken from arg[0], if it exists, or from self.parameters.
            
            returns z, rho(z) - the distance from the top interface and the SLD at that point
            
        """
        if test_parameters is None:
            test_parameters = self.parameters
            
        if 'points' in kwds and kwds['points'] is not None:
            return points, sld_profile(test_parameters, points)
        
        if not int(test_parameters[0]):
            zstart= -5 - 4 * math.fabs(test_parameters[7])
        else:
            zstart= -5 - 4 * math.fabs(test_parameters[11])
        
        temp = 0
        if not int(test_parameters[0]):
            zend = 5 + 4 * math.fabs(test_parameters[7])
        else:
            for ii in xrange(int(test_parameters[0])):
                temp += math.fabs(test_parameters[4 * ii + 8])
            zend = 5 + temp + 4 * math.fabs(test_parameters[7])
            
        points = np.linspace(zstart, zend, num = 500)
        
        return points, sld_profile(test_parameters, points)
    
    def progress(self, iterations, convergence, chi2, *args):
        return True
        

def costfunction_logR_noweight(modeldata, ydata, edata, test_parameters):
    return np.sum(np.power(np.log10(modeldata) - np.log10(ydata), 2))
    
def costfunction_logR_weight(modeldata, ydata, edata, test_parameters):
    intensity, sd = EP.EPlog10(ydata, edata)
    return  np.sum(np.power((intensity - np.log10(modeldata)) / sd, 2))

    
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
