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
    def __init__(self, xdata, ydata, edata, parameters, args = (), **kwds):

        super(line, self).__init__(xdata,
                                     ydata,
                                      edata,
                                       None,
                                        parameters,
                                         args = args, **kwds)

    def model(self, parameters, args = ()):
        return parameters[0] + self.xdata * parameters[1]                                                     
                    
        