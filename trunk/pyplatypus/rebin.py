from __future__ import division
import numpy as np


def rebin(x_init, y_init, s_init, x_rebin, verbose = False):
    """
    Rebins histogrammed data into boundaries set by x_rebin.
    returns (rebin, rebinSD)
    
    The data should be sorted beforehand.
    
    it uses linear interpolation to work out the proportions of which original cells should be placed
    in the new bin boundaries.
    
    Precision will normally be lost.  
    It does not make sense to rebin to smaller bin boundaries.
    
    when we calculate the standard deviation on the intensity carry the variance through the calculation
    and convert to SD at the end.
    """
    var_init = (np.copy(s_init))**2
    
    W_rebin = np.zeros(len(x_rebin) - 1)
    W_rebinSD = np.zeros(len(x_rebin) - 1)

    positions = np.interp(x_rebin, x_init, np.arange(np.size(x_init), dtype = 'float64'), left = 0., right = len(x_init) - 1)
    if verbose:
        print positions, x_rebin, x_init
        
    cumsum = np.cumsum(y_init, dtype='float64')
    cumsumvar = np.cumsum(var_init, dtype = 'float64')
    
    cumsum = np.insert(cumsum, 0, np.zeros(1))
    cumsumvar = np.insert(cumsumvar, 0, np.zeros(1))
    
    #	values = utility.searchinterp(cumsum, positions)
    values = np.interp(positions, np.arange(len(cumsumvar), dtype = 'float64'), cumsum)
    W_rebin = values[1:] - values[:-1]
    
    #	values = utility.searchinterp(cumsumvar, positions)
    values = np.interp(positions, np.arange(len(cumsumvar), dtype = 'float64'), cumsumvar)
    W_rebinSD = values[1:] - values[:-1]
    
    celloc = np.where(np.ceil(positions) - 1 < 0, 0, np.ceil(positions) - 1)
    celloc = celloc.astype('int')
    
    W_rebinSD -=  (np.ceil(positions[1:]) - positions[1:]) * var_init[celloc[1:]] * (1 - np.ceil(positions[1:]) + positions[1:])
    W_rebinSD -=  (np.ceil(positions[:-1]) - positions[:-1]) * var_init[celloc[:-1]] * (1 - np.ceil(positions[:-1]) + positions[:-1])

    assert not np.less(W_rebinSD, 0).any()          
    return W_rebin, np.sqrt(W_rebinSD)
	
def rebin_test():
    a = np.array([0,1,2,3,4,5,6.6])
    b = np.array([2,3,4,5,5,4])
    c = np.sqrt(b)
    d = np.array([-0.2,0.8,1.5,2.5,3.5,8])
    e,f = rebin(a,b,c,d, verbose=True)
#    print e, f, '\n'
    
    a = np.arange(10.)
    b = np.arange(11.)
    c = np.array([0,1,2,3,4,8,11])

    e,f = rebin(b,a,np.sqrt(a),c)
#    print e, f

	
def rebin_Q(qq, rr, dr, dq, lowerQ = 0.005, upperQ = 0.4, rebinpercent = 4):
    """
    this function rebins a set of R vs Q data given a rebin percentage.
    it is designed to replace rebinning the wavelength spectrum which can result in twice as many points in the overlap region.
    However, the background subtraction is currently done on rebinned data. So if you don't rebin at the start the  subtraction
    isn't as good.
    """
    rebin =  1 + (rebinpercent / 100.)
    stepsize = np.log10(rebin)
    numsteps = np.log10(upperQ / lowerQ) // stepsize
    
    W_q_rebin = np.zeros(numsteps + 1)
    W_R_rebin = np.zeros(numsteps + 1)
    W_E_rebin = np.zeros(numsteps + 1)
    W_dq_rebin = np.zeros(numsteps + 1)
    Q_sw = np.zeros(numsteps + 1)
    I_sw = np.zeros(numsteps + 1)
    
    W_q_rebinHIST = np.logspace(np.log10(lowerQ), np.log10(upperQ), num = numsteps + 2)
    
    weight = 1 / (dr**2)
     
    binnum = np.interp(qq, W_q_rebinHIST, np.arange(np.size(W_q_rebinHIST), dtype = 'float64'), left = np.nan, right = np.nan)

    binnum = (np.ceil(binnum) - 1).astype('int')
    binnum = np.reshape(binnum, (len(binnum), ))
    binnum = np.where(binnum == -1, 0, binnum)
	
	#only those points that will fit in the new histogram
	#binnum = np.extract(np.isfinite(binnum), binnum)    
    for index, val in np.ndenumerate(binnum):
    	if np.isnan(val):
    		continue
        W_R_rebin[val] += rr[(index)] * weight[(index)]		
        W_q_rebin[val] += qq[(index)] * weight[(index)]
        W_dq_rebin[val] += dq[(index)] * weight[(index)]
        Q_sw[val] += weight[(index)]
        I_sw[val] += weight[(index)]

    W_R_rebin /= I_sw
    W_q_rebin /= Q_sw
    W_E_rebin = np.sqrt(1/I_sw)
    W_dq_rebin /= Q_sw

    d = np.where(np.isnan(W_q_rebin) | np.isinf(W_q_rebin))
    
    W_q_rebin = np.delete(W_q_rebin, d[0])
    W_R_rebin = np.delete(W_R_rebin, d[0])
    W_E_rebin = np.delete(W_E_rebin, d[0])
    W_dq_rebin = np.delete(W_dq_rebin, d[0])
    return W_q_rebin, W_R_rebin, W_E_rebin, W_dq_rebin

	
def rebin2D(x_init, y_init, z_init, s_init, x_rebin, y_rebin):    
    intermed = np.zeros((np.size(x_init) - 1, np.size(y_rebin) - 1), dtype = 'float64')
    intermedSD = np.zeros((np.size(x_init) - 1, np.size(y_rebin) - 1), dtype = 'float64')
    assert not np.isnan(s_init).any()
    assert not np.less(s_init, 0).any()
    
    for ii in np.arange(len(x_init) - 1):
        intermed[ii,:], intermedSD[ii,:] = rebin(y_init, z_init[ii,:], s_init[ii,:], y_rebin)
        assert not np.isnan(intermedSD).any()
        assert not np.less(intermedSD, 0).any()

    z_rebin = np.zeros((np.size(x_rebin, 0) - 1, np.size(y_rebin, 0) - 1), dtype = 'float64')
    z_rebinSD = np.zeros(z_rebin.shape, dtype = 'float64')
    
    for ii in np.arange(len(y_rebin) - 1):
        z_rebin[:, ii], z_rebinSD[:, ii] = rebin(x_init, intermed[:, ii], intermedSD[:, ii], x_rebin, verbose=False)
        assert not np.isnan(z_rebinSD).any()

    return z_rebin, z_rebinSD