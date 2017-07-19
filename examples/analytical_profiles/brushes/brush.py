from __future__ import division
import os.path
import numpy as np

from scipy.interpolate import PchipInterpolator as Pchip
from scipy.integrate import simps, trapz

from refnx.reflect import ReflectModel, Structure, Component, SLD, Slab
from refnx.analysis import (Bounds, Parameter, Parameters,
                            possibly_create_parameter)


class FreeformVFP(Component):
    """
    """
    def __init__(self, extent, vf, dz, polymer_sld, solvent, name='',
                 gamma=None, left_slabs=(), right_slabs=(),
                 interpolator=Pchip, zgrad=True, microslab_max_thickness=2):
        """
        Parameters
        ----------
        """
        self.name = name

        if isinstance(polymer_sld, SLD):
            self.polymer_sld = polymer_sld
        else:
            self.polymer_sld = SLD(polymer_sld)

        self.left_slabs = [slab for slab in left_slabs if
                           isinstance(slab, Slab)]
        self.right_slabs = [slab for slab in right_slabs if
                            isinstance(slab, Slab)]

        self.solvent_slab = solvent
        self.microslab_max_thickness = microslab_max_thickness

        self.extent = (
            possibly_create_parameter(extent,
                                      name='%s - spline extent' % name))

        self.dz = Parameters(name='dz - spline')
        for i, z in enumerate(dz):
            p = possibly_create_parameter(
                z,
                name='%s - spline dz[%d]' % (name, i))
            p.range(0, 1)
            self.dz.append(p)

        self.vf = Parameters(name='vf - spline')
        for i, v in enumerate(vf):
            p = possibly_create_parameter(
                v,
                name='%s - spline vf[%d]' % (name, i))
            p.range(0, 1)
            self.vf.append(p)

        if len(self.vf) != len(self.dz):
            raise ValueError("dz and vs must have same number of entries")

        self.zgrad = zgrad
        self.interpolator = interpolator

        if gamma is not None:
            self.gamma = possibly_create_parameter(gamma, 'gamma')
        else:
            self.gamma = Parameter(0, 'gamma')

    def __call__(self, z):
        # calculate spline value at z
        zeds = np.cumsum(self.dz)

        # if dz's sum to more than 1, then normalise to unit interval.
        if np.sum(self.dz) > 1:
            zeds /= np.sum(self.dz)

        vf = np.array(self.vf)

        if len(self.left_slabs):
            left_end = 1 - self.left_slabs[-1].vfsolv.value
        else:
            left_end = vf[0]

        if len(self.right_slabs):
            right_end = 1 - self.right_slabs[0].vfsolv.value
        else:
            right_end = vf[-1]

        if self.zgrad:
            zeds = np.r_[-1.1, 0, zeds, 1, 2.1]
            vf = np.r_[left_end, left_end, vf, right_end, right_end]
        else:
            zeds = np.r_[0, zeds, 1]
            vf = np.r_[left_end, vf, right_end]
        print(zeds, vf)
        return self.interpolator(zeds, vf)(z / float(self.extent))

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
        p.extend([self.extent, self.dz, self.vf, self.solvent_slab,
                  self.polymer_sld, self.gamma])
        p.extend([slab.parameters for slab in self.left_slabs])
        p.extend([slab.parameters for slab in self.right_slabs])
        return p

    def lnprob(self):
        # log-probability for area under profile
        return self.gamma.lnprob(self.profile_area())

    def profile_area(self):
        """
        Calculates integrated area of volume fraction profile

        Returns
        -------
        area: integrated area of volume fraction profile
        """
        slabs = self.slabs
        areas = self.slabs[..., 0] * (1 - slabs[..., 4])
        area = np.sum(areas)

        for slab in self.left_slabs:
            area += slab[0, 0] * (1 - slab[0, 4])
        for slab in self.right_slabs:
            area += slab[0, 0] * (1 - slab[0, 4])

        return area

    @property
    def slabs(self):
        num_slabs = np.ceil(float(self.extent) / self.microslab_max_thickness)
        slab_thick = float(self.extent / num_slabs)
        slabs = np.zeros((int(num_slabs), 5))
        slabs[:, 0] = slab_thick

        # give each slab a miniscule roughness
        slabs[:, 3] = 0.5

        dist = np.cumsum(slabs[..., 0]) - 0.5 * slab_thick
        slabs[:, 1] = self.polymer_sld.real.value
        slabs[:, 2] = self.polymer_sld.imag.value
        slabs[:, 4] = 1 - self(dist)

        return slabs
