from __future__ import division
import numpy as np

def get_scaling_in_overlap(x0,y0, dy0, x1, y1, dy1):
    """
    Obtain vertical scaling factor that splices the second dataset onto the
    first.

    Parameters
    ----------
    x0: np.ndarray
        abscissae for the first dataset

    y0: np.ndarray
        y values for the first dataset
    dy0: np.ndarray
        dy (standard deviation) values for the first dataset
    x1: np.ndarray
        abscissae values for the second dataset
    y1: np.ndarray
        y values for the second dataset
    dy1: np.ndarray
        dy (standard deviation) values for the second dataset

    Returns
    -------
    (scale, dscale):
        The scaling factor and the uncertainty in scaling factor, or
        (np.nan, np.nan) if there is no overlap in abscissae values.
    """

    # the datasets should be sorted, but we may not want to sort the data
    # so make a temporary copy of the data

    sortarray = np.argsort(x0)
    tx0 = x0[sortarray]
    ty0 = y0[sortarray]
    tdy0 = dy0[sortarray]
    
    sortarray = np.argsort(x1)
    tx1 = x1[sortarray]
    ty1 = y1[sortarray]
    tdy1 = dy1[sortarray]
    
    #largest point number of x1 in overlap region
    num2 = np.interp(tx0[-1:-2:-1], tx1, np.arange(len(tx1) * 1.))
    num2 = int(np.ceil(num2[0]))

    if num2 == 0:
        return np.NaN, np.NaN

    #get scaling factor at each point of dataset 2 in the overlap region
    #get the intensity of wave1 at an overlap point
    newi = np.interp(tx1[:num2], tx0, ty0)
    newdi = np.interp(tx1[:num2], tx0, tdy0)
    
    W_scalefactor = newi / ty1[:num2]
    W_dscalefactor = W_scalefactor * np.sqrt((newdi / newi)**2
                                             + (tdy1[:num2] / ty1[:num2])**2)
    W_dscalefactor =  np.sqrt((newdi / ty1[:num2])**2
                              + ((newi * tdy1[:num2])**2) / ty1[:num2]**4)


    W_dscalefactor = 1 / (W_dscalefactor**2)
    
    num = np.sum(W_scalefactor * W_dscalefactor)
    den = np.sum(W_dscalefactor)
 
    normal = num / den
    dnormal = np.sqrt(1 / den)

    return normal, dnormal
	