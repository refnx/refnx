from __future__ import division
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.integrate as spi
import math
import pyplatypus.analysis.fitting as fitting
import pyplatypus.util.ErrorProp as EP
import pyplatypus.analysis.reflect as reflect


class line(fitting.FitObject):
    '''
        fitfunction for a straightline
        y = p[0] + p[1] * x
    '''
    def __init__(self, xdata, ydata, edata, parameters, args=(), **kwds):

        super(line, self).__init__(xdata,
                                   ydata,
                                   edata,
                                   None,
                                   parameters,
                                   args=args, **kwds)

    def model(self, parameters, *args):
        return parameters[0] + self.xdata * parameters[1]

class gauss1D(fitting.FitObject):
    '''
        fitfunction for a Gaussian
        y = p[0] + p[1] * exp(-0.5 * ((x - p[2])/p[3])**2)
    '''

    def __init__(self, xdata, ydata, edata, parameters, args=(), **kwds):

        super(gauss1D, self).__init__(xdata,
                                   ydata,
                                   edata,
                                   None,
                                   parameters,
                                   args=args, **kwds)

    def model(self, parameters, *args):
        return parameters[0] + parameters[1] * \
            np.exp(-0.5 * np.power((self.xdata - parameters[2])/parameters[3], 2))
