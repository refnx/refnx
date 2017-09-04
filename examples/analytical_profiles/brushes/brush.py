from __future__ import division
import os.path
import numpy as np

from scipy.interpolate import PchipInterpolator as Pchip
from scipy.integrate import simps

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

        if isinstance(solvent, SLD):
            self.solvent = solvent
        else:
            self.solvent = SLD(solvent)

        # left and right slabs are other areas where the same polymer can
        # reside
        self.left_slabs = [slab for slab in left_slabs if
                           isinstance(slab, Slab)]
        self.right_slabs = [slab for slab in right_slabs if
                            isinstance(slab, Slab)]

        self.microslab_max_thickness = microslab_max_thickness

        self.extent = (
            possibly_create_parameter(extent,
                                      name='%s - spline extent' % name))

        # dz are the spatial spacings of the spline knots
        self.dz = Parameters(name='dz - spline')
        for i, z in enumerate(dz):
            p = possibly_create_parameter(
                z,
                name='%s - spline dz[%d]' % (name, i))
            p.range(0, 1)
            self.dz.append(p)

        # vf are the volume fraction values of each of the spline knots
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
        """
        Calculates the volume fraction profile of the spline
        """
        zeds = np.cumsum(self.dz)

        # if dz's sum to more than 1, then normalise to unit interval.
        if np.sum(self.dz) > 1:
            zeds /= np.sum(self.dz)

        vf = np.array(self.vf)

        # use the volume fraction of the last left_slab as the initial vf of
        # the spline
        if len(self.left_slabs):
            left_end = 1 - self.left_slabs[-1].vfsolv.value
        else:
            left_end = vf[0]

        # in contrast use a vf = 0 for the last vf of
        # the spline, unless right_slabs is specified
        if len(self.right_slabs):
            right_end = 1 - self.right_slabs[0].vfsolv.value
        else:
            right_end = 0

        # do you require zero gradient at either end of the spline?
        if self.zgrad:
            zeds = np.r_[-1.1, 0, zeds, 1, 2.1]
            vf = np.r_[left_end, left_end, vf, right_end, right_end]
        else:
            zeds = np.r_[0, zeds, 1]
            vf = np.r_[left_end, vf, right_end]
        # print(zeds, vf)

        # TODO make vfp zero for z > self.extent
        vfp = self.interpolator(zeds, vf)(z / float(self.extent))
        return vfp

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
        zed, profile = self.profile()
        profile *= zed**moment
        val = simps(profile, zed)
        area = self.profile_area()
        return val / area

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.extent, self.dz, self.vf, self.solvent.parameters,
                  self.polymer_sld.parameters, self.gamma])
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
            _slabs = slab.slabs
            area += _slabs[0, 0] * (1 - _slabs[0, 4])
        for slab in self.right_slabs:
            _slabs = slab.slabs
            area += _slabs[0, 0] * (1 - _slabs[0, 4])

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

    def profile(self):
        """
        Calculates the volume fraction profile

        Returns
        -------
        z, vfp : np.ndarray
            Distance from the interface, volume fraction profile
        """
        s = Structure()
        s |= SLD(0)

        m = SLD(1.)

        for i, slab in enumerate(self.left_slabs):
            layer = m(slab.thick.value, slab.rough.value)
            if not i:
                layer.rough.value = 0
            layer.vfsolv.value = slab.vfsolv.value
            s |= layer

        polymer_slabs = self.slabs

        for i in range(np.size(polymer_slabs, 0)):
            layer = m(polymer_slabs[i, 0], polymer_slabs[i, 3])
            layer.vfsolv.value = polymer_slabs[i, -1]
            s |= layer

        for i, slab in enumerate(self.right_slabs):
            layer = m(slab.thick.value, slab.rough.value)
            layer.vfsolv.value = 1 - slab.vfsolv.value
            s |= layer

        s |= SLD(0, 0)

        # now calculate the VFP.
        total_thickness = np.sum(s.slabs[:, 0])
        zed = np.linspace(0, total_thickness, total_thickness + 1)
        # SLD profile puts a very small roughness on the interfaces with zero
        # roughness.
        zed[0] = 0.01
        z, s = s.sld_profile(z=zed)
        s[0] = s[1]

        return z, s
