from __future__ import division
import numpy as np

from scipy.interpolate import PchipInterpolator as Pchip

from refnx.reflect import Structure, Component
from refnx.analysis import (Parameter, Parameters,
                            possibly_create_parameter)


class Spline(Component):
    """
    Freeform modelling of the real part of an SLD profile using spline
    interpolation.
    """

    def __init__(self, extent, vs, dz, left, right, solvent, name='',
                 interpolator=Pchip, zgrad=True, microslab_max_thickness=2):
        """
        Parameters
        ----------
        extent : float or Parameter
            Total extent of spline region
        vs : Sequence of float/Parameter
            the real part of the SLD values of each of the knots.
        dz : Sequence of float/Parameter
            the lateral offset between successive knots.
        left : refnx.reflect.Component
            The Component to the left of this Spline region.
        right : refnx.reflect.Component
            The Component to the right of this Spline region.
        solvent : refnx.reflect.Slab
            A Slab instance representing the solvent
        name : str
            Name of component
        interpolator : scipy.interpolate Univariate Interpolator, optional
            Which scipy.interpolate Univariate Interpolator to use.
        zgrad : bool, optional
            If true then extra control knots are placed outside this spline
            with the same SLD as the materials on the left and right. With a
            monotonic interpolator this guarantees that the gradient is zero
            at either end of the interval.
        microslab_max_thickness : float
            Maximum size of the microslabs approximating the spline.

        Notes
        -----
        This spline component only generates the real part of the SLD (thereby
        assuming that the imaginary part is negligible).
        The sequence dz are the lateral offsets of the knots normalised to a
        unit interval [0, 1]. This means the cumulative sum must be less than
        one, `np.cumsum(dz)[-1] <= 1`. The reason for using lateral offsets is
        so that the knots are monotonically increasing in location. When each
        dz offset is turned into a Parameter it is given a minimum of 0.
        Thus with an extent of 500, and dz = [0.1, 0.2, 0.2], the knots will be
        at [0, 50, 150, 250, 500]. There are two extra knots for the start and
        end of the interval (disregarding the `zgrad` control knots).
        If `vs` is monotonic then the output spline will be monotonic. If `vs`
        is not monotonic then there may be regions of the spline larger or
        smaller than `left` or `right`.
        The slab representation of this component are approximated using a
        'microslab' representation of spline. The max thickness of each
        microslab is `microslab_max_thickness`.
        """
        self.name = name
        self.left_Slab = left
        self.right_Slab = right
        self.solvent_Slab = solvent
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

        self.vs = Parameters(name='vs - spline')
        for i, v in enumerate(vs):
            p = possibly_create_parameter(
                v,
                name='%s - spline vs[%d]' % (name, i))
            self.vs.append(p)

        if len(self.vs) != len(self.dz):
            raise ValueError("dz and vs must have same number of entries")

        self.zgrad = zgrad
        self.interpolator = interpolator

    def __call__(self, z):
        # calculate spline value at z
        zeds = np.cumsum(self.dz)
        vs = np.array(self.vs)

        left_sld = Structure.overall_sld(
            np.atleast_2d(self.left_Slab.slabs[-1]),
            self.solvent_Slab.slabs)[..., 1]

        right_sld = Structure.overall_sld(
            np.atleast_2d(self.right_Slab.slabs[0]),
            self.solvent_Slab.slabs)[..., 1]

        if self.zgrad:
            zeds = np.r_[-1.1, 0, zeds, 1, 2.1]
            vs = np.r_[left_sld, left_sld, vs, right_sld, right_sld]
        else:
            zeds = np.r_[0, zeds, 1]
            vs = np.r_[left_sld, vs, right_sld]
        return self.interpolator(zeds, vs)(z / float(self.extent))

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.extent, self.dz, self.vs,
                  self.left_Slab,
                  self.right_Slab,
                  self.solvent_Slab])
        return p

    def lnprob(self):
        zeds = np.cumsum(self.dz)
        if zeds[-1] > 1:
            return -np.inf

    @property
    def slabs(self):
        num_slabs = np.ceil(float(self.extent) / self.microslab_max_thickness)
        slab_thick = float(self.extent / num_slabs)
        slabs = np.zeros((int(num_slabs), 5))
        slabs[:, 0] = slab_thick

        # give each slab a miniscule roughness
        slabs[:, 3] = 0.5

        dist = np.cumsum(slabs[..., 0]) - 0.5 * slab_thick
        slabs[:, 1] = self(dist)

        return slabs
