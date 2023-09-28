import numpy as np

from scipy.interpolate import PchipInterpolator as Pchip

from refnx.reflect import Structure, Component
from refnx.analysis import Parameter, Parameters, possibly_create_parameter


EPS = np.finfo(float).eps


class Spline(Component):
    """
    Freeform modelling of the real part of an SLD profile using spline
    interpolation.

    Parameters
    ----------
    extent : float or Parameter
        Total extent of spline region
    vs : Sequence of float/Parameter
        the real part of the SLD values of each of the knots.
    dz : Sequence of float/Parameter
        the lateral offset between successive knots.
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
    unit interval [0, 1]. The reason for using lateral offsets is
    so that the knots are monotonically increasing in location. When each
    dz offset is turned into a Parameter it is given bounds in [0, 1].
    Thus with an extent of 500, and dz = [0.1, 0.2, 0.2], the knots will be
    at [0, 50, 150, 250, 500]. Notice that there are two extra knots for
    the start and end of the interval (disregarding the `zgrad` control
    knots). If ``np.sum(dz) > 1``, then the knot spacings are normalised to
    1. e.g. dz of [0.1, 0.2, 0.9] would result in knots (in the normalised
    interval) of [0, 0.0833, 0.25, 1, 1].
    If `vs` is monotonic then the output spline will be monotonic. If `vs`
    is not monotonic then there may be regions of the spline larger or
    smaller than `left` or `right`.
    The slab representation of this component are approximated using a
    'microslab' representation of spline. The max thickness of each
    microslab is `microslab_max_thickness`.

    A Spline component should not be used more than once in a given Structure.
    """

    def __init__(
        self,
        extent,
        vs,
        dz,
        name="",
        interpolator=Pchip,
        zgrad=True,
        microslab_max_thickness=1,
    ):
        super().__init__()
        self.name = name
        self.microslab_max_thickness = microslab_max_thickness

        self.extent = possibly_create_parameter(
            extent, name="%s - spline extent" % name, units="Å"
        )

        self.dz = Parameters(name="dz - spline")
        for i, z in enumerate(dz):
            p = possibly_create_parameter(z, name=f"{name} - spline dz[{i}]")
            p.range(0.0000001, 1)
            self.dz.append(p)

        self.vs = Parameters(name="vs - spline")
        for i, v in enumerate(vs):
            p = possibly_create_parameter(
                v, name=f"{name} - spline vs[{i}]", units="10**-6 Å**-2"
            )
            self.vs.append(p)

        if len(self.vs) != len(self.dz):
            raise ValueError("dz and vs must have same number of entries")

        self.zgrad = zgrad
        self.interpolator = interpolator

        self.__cached_interpolator = {
            "zeds": np.array([]),
            "vs": np.array([]),
            "interp": None,
            "extent": -1,
        }

    def __repr__(self):
        s = (
            f"Spline({self.extent!r}, {self.vs!r}, {self.dz!r},"
            f" name={self.name!r}, zgrad={self.zgrad},"
            f" microslab_max_thickness={self.microslab_max_thickness})"
        )
        return s

    def _interpolator(self, structure):
        dz = np.array(self.dz)
        zeds = np.cumsum(dz)

        # if dz's sum to more than 1, then normalise to unit interval.
        if len(zeds) and zeds[-1] > 1:
            # there may be no knots
            zeds /= zeds[-1]
            zeds = np.clip(zeds, 0, 1)

        # note - this means you shouldn't use the same Spline more than once in
        # a Component, because only the first use will be detected.
        try:
            loc = structure.index(self)
            # figure out SLDs for the bracketing Components.
            # note the use of the modulus operator. This means that if the
            # Spline is at the end, then the right most Component will be
            # assumed to be the first Component. This is to aid the use of
            # Spline in a Stack.
            left_component = structure[loc - 1]
            right_component = structure[(loc + 1) % len(structure)]
        except ValueError:
            raise ValueError(
                "Spline didn't appear to be part of a super Structure"
            )

        if isinstance(left_component, Spline) or isinstance(
            right_component, Spline
        ):
            raise ValueError(
                "Spline must be bracketed by Components that"
                " aren't Splines."
            )

        vs = np.array(self.vs)

        left_sld = structure.overall_sld(
            np.atleast_2d(left_component.slabs(structure)[-1]),
            structure.solvent,
        )[..., 1]

        right_sld = structure.overall_sld(
            np.atleast_2d(right_component.slabs(structure)[0]),
            structure.solvent,
        )[..., 1]

        if self.zgrad:
            zeds = np.concatenate([[-1.1, 0 - EPS], zeds, [1 + EPS, 2.1]])
            vs = np.concatenate([left_sld, left_sld, vs, right_sld, right_sld])
        else:
            zeds = np.concatenate([[0 - EPS], zeds, [1 + EPS]])
            vs = np.concatenate([left_sld, vs, right_sld])

        # cache the interpolator
        cache_zeds = self.__cached_interpolator["zeds"]
        cache_vs = self.__cached_interpolator["vs"]
        cache_extent = self.__cached_interpolator["extent"]

        # you don't need to recreate the interpolator
        if (
            np.array_equal(zeds, cache_zeds)
            and np.array_equal(vs, cache_vs)
            and np.equal(self.extent, cache_extent)
        ):
            return self.__cached_interpolator["interp"]
        else:
            self.__cached_interpolator["zeds"] = zeds
            self.__cached_interpolator["vs"] = vs
            self.__cached_interpolator["extent"] = float(self.extent)

        # TODO make vfp zero for z > self.extent
        interpolator = self.interpolator(zeds, vs)
        self.__cached_interpolator["interp"] = interpolator
        return interpolator

    def __call__(self, z, structure):
        """
        Calculates the spline value at z

        Parameters
        ----------
        z : float
            Distance along spline
        structure: refnx.reflect.Structure
            Structure hosting this Component

        Returns
        -------
        sld : float
            Real part of SLD
        """
        interpolator = self._interpolator(structure)
        vs = interpolator(z / float(self.extent))
        return vs

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.extent, self.dz, self.vs])
        return p

    def logp(self):
        return 0

    def slabs(self, structure=None):
        """
        Slab representation of the spline, as an array

        Parameters
        ----------
        structure : refnx.reflect.Structure
            The Structure hosting this Component
        """
        if structure is None:
            raise ValueError("Spline.slabs() requires a valid Structure")

        num_slabs = np.ceil(float(self.extent) / self.microslab_max_thickness)
        slab_thick = float(self.extent / num_slabs)
        slabs = np.zeros((int(num_slabs), 5))
        slabs[:, 0] = slab_thick

        # give last slab a miniscule roughness so it doesn't get contracted
        slabs[-1:, 3] = 0.5

        dist = np.cumsum(slabs[..., 0]) - 0.5 * slab_thick
        slabs[:, 1] = self(dist, structure)

        return slabs
