import numpy as np
from refnx.analysis import (
    possibly_create_parameter,
    Parameter,
    is_parameter,
    Parameters,
)
from refnx.reflect import Component, Structure, SLD


class FunctionalForm(Component):
    """
    Component used to describe an analytic SLD profile.

    Parameters
    ----------
    extent : Parameter or float
        The total extent of the functional region
    profile : callable
        callable returning the SLD of the functional form. Has the signature
        ``profile(z, extent, left_sld, right_sld, **kwds)`` where ``z`` is an
        array specifying the distances at which the functional form should be
        calculated, ``extent`` is a float specifying the total extent of the
        Component, ``left_sld`` and ``right_sld`` are the complex SLDs of the
        Components preceding and following this Component. ``kwds`` are extra
        parameters used to calculate the shape of the functional form.
        The function should return a tuple, ``(slds, vfsolv)`` where ``slds``
        is an array (possibly of type np.complex128) of shape `(len(z),)`.
        Similarly, ``vfsolv`` is an array of shape `(len(z))` specifying the
        volume fraction of solvent at each distance in ``z``. If ``vfsolv`` is
        not necessary then return ``(slds, None)``.
    name : str
        Name of component
    microslab_max_thickness : float, optional
        Thickness of microslicing of spline for reflectivity calculation
    kwds : dict
        Named keywords that supply extra Parameters to the ``profile`` callable.
        These parameters are passed numerically, not as Parameter objects.

    Examples
    --------
    A linear ramp. Note that the `dummy_param` is not actually used anywhere.

    >>> def line(z, extent, left_sld, right_sld, dummy_param=None):
    ...     grad = (right_sld - left_sld) / extent
    ...     intercept = left_sld
    ...     # we don't calculate the volume fraction of solvent
    ...     return z*grad*dummy_param + intercept, None

    >>> si = SLD(2.07)
    >>> d2o = SLD(6.36)
    >>> p = Parameter(1)

    >>> form = FunctionalForm(100, line, dummy_param=p)
    >>> s = si | form | d2o(0, 3)

    A quadratic example that goes through the two end points

    >>> def quadratic(z, extent, left_sld, right_sld, x=None, y=None):
    ...     res = np.polyfit(
    ...         [0., x, extent],
    ...         [np.real(left_sld), y, np.real(right_sld)],
    ...         deg=2
    ...     )
    ...     return np.polyval(res, z), None

    >>> si = SLD(2.07)
    >>> d2o = SLD(6.36)
    >>> x = Parameter(4.)
    >>> y = Parameter(5.)
    >>> quad = FunctionalForm(100., quadratic, x=x, y=y)
    >>> s = si | quad | d2o(0, 3)

    """

    def __init__(
        self, extent, profile, name=None, microslab_max_thickness=1, **kwds
    ):
        super(FunctionalForm, self).__init__(name=name)
        self.profile = profile
        self.other_params = {k: v for k, v in kwds.items() if is_parameter(v)}

        self.extent = possibly_create_parameter(extent)
        self.microslab_max_thickness = microslab_max_thickness

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.extent])
        p.extend(list(self.other_params.values()))
        return p

    def slabs(self, structure=None):
        assert (
            structure is not None
        ), "In order to calculate slab representation please provide a Structure"

        try:
            loc = structure.index(self)
            # figure out SLDs for the bracketing Components.
            left_component = structure[loc - 1]
            right_component = structure[(loc + 1) % len(structure)]
        except ValueError:
            raise ValueError(
                "FunctionalForm didn't appear to be part of a super structure"
            )

        left_slab = structure.overall_sld(
            np.atleast_2d(left_component.slabs(structure)[-1]),
            structure.solvent,
        )
        left_sld = complex(left_slab[..., 1][0], left_slab[..., 2][0])

        right_slab = structure.overall_sld(
            np.atleast_2d(right_component.slabs(structure)[0]),
            structure.solvent,
        )
        right_sld = complex(right_slab[..., 1][0], right_slab[..., 2][0])

        num_slabs = int(
            np.ceil(float(self.extent) / self.microslab_max_thickness)
        )
        slab_thick = self.extent.value / num_slabs

        slabs = np.zeros((num_slabs, 5))
        slabs[..., 0] = slab_thick

        dist = np.cumsum(slabs[..., 0]) - 0.5 * slab_thick

        pars = {k: float(v) for k, v in self.other_params.items()}
        res, vfsolv = self.profile(
            dist, self.extent.value, left_sld, right_sld, **pars
        )

        slabs[..., 1] = np.real(res)
        slabs[..., 2] = np.imag(res)

        if vfsolv is not None:
            slabs[..., 4] = np.clip(vfsolv, 0, 1)

        return slabs
