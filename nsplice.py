import numpy as np

def getScalingInOverlap(qq1,rr1, dr1, qq2, rr2, dr2):
    """
    Get the vertical scaling factor that would splice the second dataset onto the first.
    returns the scaling factor and the uncertainty in scaling factor
    """
    #sort the abscissae
    sortarray = np.argsort(qq1)
    qq1 = qq1[sortarray]
    rr1 = rr1[sortarray]
    dr1 = dr1[sortarray]
    
    sortarray = np.argsort(qq2)
    qq2 = qq2[sortarray]
    rr2 = rr2[sortarray]
    dr2 = dr2[sortarray]
    
    #largest point number of qq2 in overlap region
    num2 = np.interp(qq1[-1:-2:-1], qq2, np.arange(len(qq2) * 1.))
    
    if np.size(num2) == 0:
        return np.NaN, np.NaN
    num2 = int(num2[0])
    
    #get scaling factor at each point of dataset 2 in the overlap region
    #get the intensity of wave1 at an overlap point
    #print qq1.shape, rr1.shape, dr1.shape
    newi = np.interp(qq2[:num2], qq1, rr1)
    newdi = np.interp(qq2[:num2], qq1, dr1)
    
    W_scalefactor = newi / rr2[:num2]
    W_dscalefactor = W_scalefactor * np.sqrt((newdi / newi)**2 + (dr2[:num2] / rr2[:num2])**2)
    W_dscalefactor = 1 / (W_dscalefactor**2)
    
    num = np.sum(W_scalefactor * W_dscalefactor)
    den = np.sum(W_dscalefactor)
 
    normal = num / den
    dnormal = np.sqrt(1/den)

    return normal, dnormal
	