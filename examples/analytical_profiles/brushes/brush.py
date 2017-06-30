from __future__ import division
import os.path
import numpy as np

from scipy.interpolate import Akima1DInterpolator, InterpolatedUnivariateSpline
from scipy.integrate import simps, trapz

from refnx.reflect import ReflectModel, Structure, Component, SLD, Slab
from refnx.analysis import (Bounds, Parameter, Parameters,
                            possibly_create_parameter)


class Spline(Component):
    def __init__(self, extent, phi, dz, name=''):
        self.name = name
        self.extent = (
            possibly_create_parameter(extent,
                                      name='%s - spline extent' % name))

        self.dz = Parameters(name='dz')
        for z in dz:
            p = Parameter(z)
            p.range(0, 1)
            self.dz.append(p)

        self.phi = Parameters(name='phi')
        for v in phi:
            p = Parameter(v)
            p.range(0, 1)
            self.phi.append(p)

        if len(self.phi) != len(self.dz):
            raise ValueError("dz and phi must have same number of entries")

    def __call__(self, z):
        # calculate spline value at z
        pass

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.append(self.extent)
        p.append(self.dz)
        p.append(self.phi)
        return p


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
