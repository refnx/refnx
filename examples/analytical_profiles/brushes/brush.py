from __future__ import division
import os.path
import numpy as np

from scipy.interpolate import PchipInterpolator as Pchip
from scipy.integrate import simps, trapz

from refnx.reflect import ReflectModel, Structure, Component, SLD, Slab
from refnx.analysis import (Bounds, Parameter, Parameters,
                            possibly_create_parameter)



class Freeform_VFP(Component):
    """
    """
    def __init__(self, polymer_SLD, spline, slabs=None, gamma=None):
        """
        Parameters
        ----------
        """
        super(Component, self).__init__()
        self.gamma = gamma
        self.pre_slabs

    def moment(self, moment=1):
        """
        Calculates the n'th moment of the profile

        Parameters
        ----------
        moment : int
            order of moment to be calculated

        Returns
        -------
        moment : float
            n'th moment
        """
        # points, profile = self.vol_fraction(params)
        # profile *= points**moment
        # val = simps(profile, points)
        # area = self.vfp_area(params)
        # return val / area
        pass

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([component.parameters for component in self.components])
        return p

    def lnprob(self):
        # log-probability for area under profile
        if isinstance(self.gamma, Bounds):
            return self.gamma.lnprob(self.profile_area())
        else:
            return 0

    def profile_area(self):
        """
        Calculates integrated area of volume fraction profile

        Returns
        -------
        area: integrated area of volume fraction profile
        """
        slabs = self.slabs
        areas = self.slabs[..., 0] * (1 - slabs[..., 4])
        return np.sum(areas)

    @property
    def slabs(self):
        pass
