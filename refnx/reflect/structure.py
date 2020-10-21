"""
refnx is distributed under the following license:

Copyright (c) 2015 A. R. J. Nelson, ANSTO

Permission to use and redistribute the source code or binary forms of this
software and its documentation, with or without modification is hereby
granted provided that the above notice of copyright, these terms of use,
and the disclaimer of warranty below appear in the source code and
documentation, and that none of the names of above institutions or
authors appear in advertising or endorsement of works derived from this
software without specific prior written permission from all parties.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THIS SOFTWARE.

"""
# -*- coding: utf-8 -*-

from collections import UserList
import numbers
import operator

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

try:
    from refnx.reflect import _creflect as refcalc
except ImportError:
    from refnx.reflect import _reflect as refcalc

from refnx._lib import flatten
from refnx.analysis import Parameters, Parameter, possibly_create_parameter
from refnx.reflect.interface import Interface, Erf, Step
from refnx.reflect.reflect_model import get_reflect_backend

# contracting the SLD profile can greatly speed a reflectivity calculation up.
contract_by_area = refcalc._contract_by_area


class Structure(UserList):
    """
    Represents the interfacial Structure of a reflectometry sample.
    Successive Components are added to the Structure to construct the
    interface.

    Parameters
    ----------
    components : sequence
        A sequence of Components to initialise the Structure.
    name : str
        Name of this structure
    solvent : refnx.reflect.Scatterer
        Specifies the scattering length density used for solvation. If no
        solvent is specified then the SLD of the solvent is assumed to be
        the SLD of `Structure[-1].slabs()[-1]` (after any possible slab order
        reversal).
    reverse_structure : bool
        If `Structure.reverse_structure` is `True` then the slab
        representation produced by `Structure.slabs` is reversed. The sld
        profile and calculated reflectivity will correspond to this
        reversed structure.
    contract : float
        If contract > 0 then an attempt to contract/shrink the slab
        representation is made. Use larger values for coarser
        profiles (and vice versa). A typical starting value to try might
        be 1.0.

    Notes
    -----
    If `Structure.reverse_structure is True` then the slab representation
    order is reversed.
    If no solvent is specified then the volume fraction of solvent in each of
    the Components is *assumed* to have the scattering length density of
    `Structure[-1].slabs()[-1]` after any possible slab order reversal. This
    slab corresponds to the scattering length density of the semi-infinite
    backing medium.
    Normally the reflectivity will be calculated using the Nevot-Croce
    approximation for Gaussian roughness between different layers. However, if
    individual components have non-Gaussian roughness (e.g. Tanh), then the
    overall reflectivity and SLD profile are calculated by micro-slicing.
    Micro-slicing involves calculating the specific SLD profile, dividing it
    up into small-slabs, and calculating the reflectivity from those. This
    normally takes much longer than the Nevot-Croce approximation. To speed
    the calculation up the `Structure.contract` property can be used.
    Contracting too far may mask the subtle differences between different
    roughness types.
    The profile contraction specified by this property can greatly improve
    calculation time for Structures created with micro-slicing. If you use
    this option it is recommended to check the reflectivity signal with and
    without contraction to ensure they are comparable.

    Example
    -------

    >>> from refnx.reflect import SLD, Linear, Tanh, Interface
    >>> # make the materials
    >>> air = SLD(0, 0)
    >>> # overall SLD of polymer is (1.0 + 0.001j) x 10**-6 A**-2
    >>> polymer = SLD(1.0 + 0.0001j)
    >>> si = SLD(2.07)
    >>> # Make the structure, s, from slabs.
    >>> # The polymer slab has a thickness of 200 A and a air/polymer roughness
    >>> # of 4 A.
    >>> s = air(0, 0) | polymer(200, 4) | si(0, 3)

    Use Linear roughness between air and polymer (rather than default Gaussian
    roughness). Use Tanh roughness between si and polymer.
    If non-default roughness is used then the reflectivity is calculated via
    micro-slicing - set the `contract` property to speed the calculation up.

    >>> s[1].interfaces = Linear()
    >>> s[2].interfaces = Tanh()
    >>> s.contract = 0.5

    Create a user defined interfacial roughness based on the cumulative
    distribution function (CDF) of a Cauchy.

    >>> from scipy.stats import cauchy
    >>> class Cauchy(Interface):
    ...     def __call__(self, x, loc=0, scale=1):
    ...         return cauchy.cdf(x, loc=loc, scale=scale)
    >>>
    >>> c = Cauchy()
    >>> s[1].interfaces = c

    """

    def __init__(
        self,
        components=(),
        name="",
        solvent=None,
        reverse_structure=False,
        contract=0,
    ):
        super(Structure, self).__init__()
        self._name = name
        self._solvent = solvent

        self._reverse_structure = bool(reverse_structure)
        #: **float** if contract > 0 then an attempt to contract/shrink the
        #: slab representation is made. Use larger values for coarser profiles
        #: (and vice versa). A typical starting value to try might be 1.0.
        self.contract = contract

        # if you provide a list of components to start with, then initialise
        # the structure from that
        self.data = [c for c in components if isinstance(c, Component)]

    def __copy__(self):
        s = Structure(name=self.name, solvent=self._solvent)
        s.data = self.data.copy()
        return s

    def __setitem__(self, i, v):
        self.data[i] = v

    def __str__(self):
        s = list()
        s.append("{:_>80}".format(""))
        s.append("Structure: {0: ^15}".format(str(self.name)))
        s.append("solvent: {0}".format(repr(self._solvent)))
        s.append("reverse structure: {0}".format(str(self.reverse_structure)))
        s.append("contract: {0}\n".format(str(self.contract)))

        for component in self:
            s.append(str(component))

        return "\n".join(s)

    def __repr__(self):
        return (
            "Structure(components={data!r},"
            " name={_name!r},"
            " solvent={_solvent!r},"
            " reverse_structure={_reverse_structure},"
            " contract={contract})".format(**self.__dict__)
        )

    def append(self, item):
        """
        Append a :class:`Component` to the Structure.

        Parameters
        ----------
        item: refnx.reflect.Component
            The component to be added.
        """
        if isinstance(item, Scatterer):
            self.append(item())
            return

        if not isinstance(item, Component):
            raise ValueError(
                "You can only add Component objects to a" " structure"
            )
        super(Structure, self).append(item)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def solvent(self):
        if self._solvent is None:
            if not self.reverse_structure:
                solv_slab = self[-1].slabs(self)
            else:
                solv_slab = self[0].slabs(self)
            return SLD(complex(solv_slab[-1, 1], solv_slab[-1, 2]))
        else:
            return self._solvent

    @solvent.setter
    def solvent(self, sld):
        if sld is None:
            self._solvent = None
        elif isinstance(sld, Scatterer):
            # don't make a new SLD object, use its reference
            self._solvent = sld
        else:
            solv = SLD(sld)
            self._solvent = solv

    @property
    def reverse_structure(self):
        """
        **bool**  if `True` then the slab representation produced by
        :meth:`Structure.slabs` is reversed. The sld profile and calculated
        reflectivity will correspond to this reversed structure.
        """
        return bool(self._reverse_structure)

    @reverse_structure.setter
    def reverse_structure(self, reverse_structure):
        self._reverse_structure = reverse_structure

    def slabs(self, **kwds):
        r"""

        Returns
        -------
        slabs : :class:`np.ndarray`
            Slab representation of this structure.
            Has shape (N, 5).

            - slab[N, 0]
               thickness of layer N
            - slab[N, 1]
               *overall* SLD.real of layer N (material AND solvent)
            - slab[N, 2]
               *overall* SLD.imag of layer N (material AND solvent)
            - slab[N, 3]
               roughness between layer N and N-1
            - slab[N, 4]
               volume fraction of solvent in layer N.

        Notes
        -----
        If `Structure.reversed is True` then the slab representation order is
        reversed. The slab order is reversed before the solvation calculation
        is done. I.e. if `Structure.solvent == 'backing'` and
        `Structure.reversed is True` then the material that solvates the system
        is the component in `Structure[0]`, which corresponds to
        `Structure.slab[-1]`.

        """
        if not len(self):
            return None

        if not (
            isinstance(self.data[-1], Slab) and isinstance(self.data[0], Slab)
        ):
            raise ValueError(
                "The first and last Components in a Structure"
                " need to be Slabs"
            )

        # Each layer can be given a different type of roughness profile
        # that defines transition between successive layers.
        # The default interface is specified by None (= Gaussian roughness)
        interfaces = flatten(self.interfaces)
        if all([i is None for i in interfaces]):
            # if all the interfaces are Gaussian, then simply concatenate
            # the default slabs property of each component.
            sl = [c.slabs(structure=self) for c in self.components]

            try:
                slabs = np.concatenate(sl)
            except ValueError:
                # some of slabs may be None. np can't concatenate arr and None
                slabs = np.concatenate([s for s in sl if s is not None])
        else:
            # there is a non-default interfacial roughness, create a microslab
            # representation
            slabs = self._micro_slabs()

        # if the slab representation needs to be reversed.
        reverse = self.reverse_structure
        if reverse:
            roughnesses = slabs[1:, 3]
            slabs = np.flipud(slabs)
            slabs[1:, 3] = roughnesses[::-1]
            slabs[0, 3] = 0.0

        if np.any(slabs[:, 4] > 0):
            # overall SLD is a weighted average of the vfs and slds
            # accessing self.solvent leads to overhead from object
            # creation.
            if self._solvent is not None:
                solv = self._solvent
            else:
                # we should always choose the solvating material to be the last
                # slab. If the structure is not reversed then you want the last
                # slab. If the structure is reversed then you should want to
                # use the first slab, but the code block above reverses the
                # slab order, so we still want the last one
                solv = complex(slabs[-1, 1], slabs[-1, 2])

            slabs[1:-1] = self.overall_sld(slabs[1:-1], solv)

        if self.contract > 0:
            return contract_by_area(slabs, self.contract)
        else:
            return slabs

    def _micro_slabs(self, slice_size=0.5):
        """
        Creates a microslab representation of the Structure.

        Parameters
        ----------
        slice_size : float
            Thickness of each slab in the micro-slab representation

        Returns
        -------
        micro_slabs : np.ndarray
            The micro-slab representation of the model. See the
            `Structure.slabs` method for a description of the array.
        """
        # solvate the slabs from each component
        sl = [c.slabs(structure=self) for c in self.components]
        total_slabs = np.concatenate(sl)
        total_slabs[1:-1] = self.overall_sld(total_slabs[1:-1], self.solvent)

        total_slabs[:, 0] = np.fabs(total_slabs[:, 0])
        total_slabs[:, 3] = np.fabs(total_slabs[:, 3])

        # interfaces between all the slabs
        _interfaces = self.interfaces
        erf_interface = Erf()
        i = 0
        # the default Interface is None.
        # The Component.interfaces property may not have the same length as the
        # Component.slabs. Expand it so it matches the number of slabs,
        # otherwise the calculation of microslabs fails.
        for _interface, _slabs in zip(_interfaces, sl):
            if _interface is None or isinstance(_interface, Interface):
                f = _interface or erf_interface
                _interfaces[i] = [f] * len(_slabs)
            i += 1

        _interfaces = list(flatten(_interfaces))
        _interfaces = [erf_interface if i is None else i for i in _interfaces]

        # distance of each interface from the fronting interface
        dist = np.cumsum(total_slabs[:-1, 0])

        # workout how much space the SLD profile should encompass
        zstart = -5.0 - 8 * total_slabs[1, 3]
        zend = 5.0 + dist[-1] + 8 * total_slabs[-1, 3]
        nsteps = int((zend - zstart) / slice_size + 1)
        zed = np.linspace(zstart, zend, num=nsteps)

        # the output arrays
        sld = np.ones_like(zed, dtype=float) * total_slabs[0, 1]
        isld = np.ones_like(zed, dtype=float) * total_slabs[0, 2]

        # work out the step in SLD at an interface
        delta_rho = total_slabs[1:, 1] - total_slabs[:-1, 1]
        delta_irho = total_slabs[1:, 2] - total_slabs[:-1, 2]

        # the RMS roughness of each step
        sigma = total_slabs[1:, 3]
        step = Step()

        # accumulate the SLD of each step.
        for i in range(len(total_slabs) - 1):
            f = _interfaces[i + 1]
            if sigma[i] == 0:
                f = step

            p = f(zed, scale=sigma[i], loc=dist[i])
            sld += delta_rho[i] * p
            isld += delta_irho[i] * p

        sld[0] = total_slabs[0, 1]
        isld[0] = total_slabs[0, 2]
        sld[-1] = total_slabs[-1, 1]
        isld[-1] = total_slabs[-1, 2]

        micro_slabs = np.zeros((len(zed), 5), float)
        micro_slabs[:, 0] = zed[1] - zed[0]
        micro_slabs[:, 1] = sld
        micro_slabs[:, 2] = isld

        return micro_slabs

    @property
    def interfaces(self):
        """
        A nested list containing the interfacial roughness types for each of
        the `Component`s.
        `len(Structure.interfaces) == len(Structure.components)`
        """
        return [c.interfaces for c in self.components]

    @staticmethod
    def overall_sld(slabs, solvent):
        """
        Performs a volume fraction weighted average of the material SLD in a
        layer and the solvent in a layer.

        Parameters
        ----------
        slabs : np.ndarray
            Slab representation of the layers to be averaged.
        solvent : complex or reflect.Scatterer
            SLD of solvating material.

        Returns
        -------
        averaged_slabs : np.ndarray
            the averaged slabs.
        """
        solv = solvent
        if isinstance(solvent, Scatterer):
            solv = complex(solvent)

        slabs[..., 1:3] *= (1 - slabs[..., 4])[..., np.newaxis]
        slabs[..., 1] += solv.real * slabs[..., 4]
        slabs[..., 2] += solv.imag * slabs[..., 4]
        return slabs

    def reflectivity(self, q, threads=0):
        """
        Calculate theoretical reflectivity of this structure

        Parameters
        ----------
        q : array-like
            Q values (Angstrom**-1) for evaluation
        threads : int, optional
            Specifies the number of threads for parallel calculation. This
            option is only applicable if you are using the ``_creflect``
            module. The option is ignored if using the pure python calculator,
            ``_reflect``. If `threads == 0` then all available processors are
            used.

        Notes
        -----
        Normally the reflectivity will be calculated using the Nevot-Croce
        approximation for Gaussian roughness between different layers. However,
        if individual components have non-Gaussian roughness (e.g. Tanh), then
        the overall reflectivity and SLD profile are calculated by
        micro-slicing. Micro-slicing involves calculating the specific SLD
        profile, dividing it up into small-slabs, and calculating the
        reflectivity from those. This normally takes much longer than the
        Nevot-Croce approximation. To speed the calculation up the
        `Structure.contract` property can be used.
        """
        abeles = get_reflect_backend()
        return abeles(q, self.slabs()[..., :4], threads=threads)

    def sld_profile(self, z=None, align=0):
        """
        Calculates an SLD profile, as a function of distance through the
        interface.

        Parameters
        ----------
        z : float
            Interfacial distance (Angstrom) measured from interface between the
            fronting medium and the first layer.
        align: int, optional
            Places a specified interface in the slab representation of a
            Structure at z = 0. Python indexing is allowed, e.g. supplying -1
            will place the backing medium at z = 0.

        Returns
        -------
        sld : float
            Scattering length density / 1e-6 Angstrom**-2

        Notes
        -----
        This can be called in vectorised fashion.
        """
        slabs = self.slabs()
        if (
            (slabs is None)
            or (len(slabs) < 2)
            or (not isinstance(self.data[0], Slab))
            or (not isinstance(self.data[-1], Slab))
        ):
            raise ValueError(
                "Structure requires fronting and backing"
                " Slabs in order to calculate."
            )

        zed, sld = sld_profile(slabs, z)

        offset = 0
        if align != 0:
            align = int(align)
            if align >= len(slabs) - 1 or align < -1 * len(slabs):
                raise RuntimeError(
                    "abs(align) has to be less than " "len(slabs) - 1"
                )
            # to figure out the offset you need to know the cumulative distance
            # to the interface
            slabs[0, 0] = slabs[-1, 0] = 0.0
            if align >= 0:
                offset = np.sum(slabs[: align + 1, 0])
            else:
                offset = np.sum(slabs[:align, 0])

        return zed - offset, sld

    def __ior__(self, other):
        """
        Build a structure by `IOR`'ing Structures/Components/SLDs.

        Parameters
        ----------
        other: :class:`Structure`, :class:`Component`, :class:`SLD`
            The object to add to the structure.

        Examples
        --------

        >>> air = SLD(0, name='air')
        >>> sio2 = SLD(3.47, name='SiO2')
        >>> si = SLD(2.07, name='Si')
        >>> structure = air | sio2(20, 3)
        >>> structure |= si(0, 4)

        """
        # self |= other
        if isinstance(other, Component):
            self.append(other)
        elif isinstance(other, Structure):
            self.extend(other.data)
        elif isinstance(other, Scatterer):
            slab = other(0, 0)
            self.append(slab)
        else:
            raise ValueError()

        return self

    def __or__(self, other):
        """
        Build a structure by `OR`'ing Structures/Components/SLDs.

        Parameters
        ----------
        other: :class:`Structure`, :class:`Component`, :class:`SLD`
            The object to add to the structure.

        Examples
        --------

        >>> air = SLD(0, name='air')
        >>> sio2 = SLD(3.47, name='SiO2')
        >>> si = SLD(2.07, name='Si')
        >>> structure = Structure()
        >>> structure = air | sio2(20, 3) | si(0, 3)

        """
        # c = self | other
        p = Structure()
        p |= self
        p |= other
        return p

    @property
    def components(self):
        """
        The list of components in the sample.
        """
        return self.data

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters`, all the parameters associated with
        this structure.

        """
        p = Parameters(name="Structure - {0}".format(self.name))
        p.extend([component.parameters for component in self.components])
        if self._solvent is not None:
            p.append(self.solvent.parameters)
        return p

    def logp(self):
        """
        log-probability for the interfacial structure. Note that if a given
        component is present more than once in a Structure then it's log-prob
        will be counted twice.

        Returns
        -------
        logp : float
            log-prior for the Structure.
        """
        logp = 0
        for component in self.components:
            logp += component.logp()

        return logp

    def plot(self, pvals=None, samples=0, fig=None, align=0):
        """
        Plot the structure.

        Requires matplotlib be installed.

        Parameters
        ----------
        pvals : np.ndarray, optional
            Numeric values for the Parameter's that are varying
        samples: number
            If this structures constituent parameters have been sampled, how
            many samples you wish to plot on the graph.
        fig: Figure instance, optional
            If `fig` is not supplied then a new figure is created. Otherwise
            the graph is created on the current axes on the supplied figure.
        align: int, optional
            Aligns the plotted structures around a specified interface in the
            slab representation of a Structure. This interface will appear at
            z = 0 in the sld plot. Note that Components can consist of more
            than a single slab, so some thought is required if the interface to
            be aligned around lies in the middle of a Component. Python
            indexing is allowed, e.g. supplying -1 will align at the backing
            medium.

        Returns
        -------
        fig, ax : :class:`matplotlib.Figure`, :class:`matplotlib.Axes`
          `matplotlib` figure and axes objects.

        """
        import matplotlib.pyplot as plt

        params = self.parameters

        if pvals is not None:
            params.pvals = pvals

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = fig.gca()

        if samples > 0:
            saved_params = np.array(params)
            # Get a number of chains, chosen randomly, and plot the model.
            for pvec in self.parameters.pgen(ngen=samples):
                params.pvals = pvec

                ax.plot(*self.sld_profile(align=align), color="k", alpha=0.01)

            # put back saved_params
            params.pvals = saved_params

        ax.plot(*self.sld_profile(align=align), color="red", zorder=20)
        ax.set_ylabel("SLD / 1e-6 $\\AA^{-2}$")
        ax.set_xlabel("z / $\\AA$")

        return fig, ax


class Scatterer(object):
    """
    Abstract base class for something that will have a scattering length
    density
    """

    def __init__(self, name=""):
        self.name = name

    def __str__(self):
        sld = complex(self)
        return "SLD = {0} x10**-6 Å**-2".format(sld)

    def __complex__(self):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError

    def __call__(self, thick=0, rough=0):
        """
        Create a :class:`Slab`.

        Parameters
        ----------
        thick: refnx.analysis.Parameter or float
            Thickness of slab in Angstrom
        rough: refnx.analysis.Parameter or float
            Roughness of slab in Angstrom

        Returns
        -------
        slab : refnx.reflect.Slab
            The newly made Slab.

        Example
        --------

        >>> # an SLD object representing Silicon Dioxide
        >>> sio2 = SLD(3.47, name='SiO2')
        >>> # create a Slab of SiO2 20 A in thickness, with a 3 A roughness
        >>> sio2_layer = sio2(20, 3)

        """
        return Slab(thick, self, rough, name=self.name)

    def __or__(self, other):
        # c = self | other
        slab = self()
        return slab | other


class SLD(Scatterer):
    """
    Object representing freely varying SLD of a material

    Parameters
    ----------
    value : float, complex, Parameter, Parameters
        Scattering length density of a material.
        Units (10**-6 Angstrom**-2)
    name : str, optional
        Name of material.

    Notes
    -----
    An SLD object can be used to create a Slab:

    >>> # an SLD object representing Silicon Dioxide
    >>> sio2 = SLD(3.47, name='SiO2')
    >>> # create a Slab of SiO2 20 A in thickness, with a 3 A roughness
    >>> sio2_layer = sio2(20, 3)

    The SLD object can also be made from a complex number, or from Parameters

    >>> sio2 = SLD(3.47+0.01j)
    >>> re = Parameter(3.47)
    >>> im = Parameter(0.01)
    >>> sio2 = SLD(re)
    >>> sio2 = SLD([re, im])
    """

    def __init__(self, value, name=""):
        super(SLD, self).__init__(name=name)

        self.imag = Parameter(0, name="%s - isld" % name)
        if isinstance(value, numbers.Real):
            self.real = Parameter(value.real, name="%s - sld" % name)
        elif isinstance(value, numbers.Complex):
            self.real = Parameter(value.real, name="%s - sld" % name)
            self.imag = Parameter(value.imag, name="%s - isld" % name)
        elif isinstance(value, SLD):
            self.real = value.real
            self.imag = value.imag
        elif isinstance(value, Parameter):
            self.real = value
        elif (
            hasattr(value, "__len__")
            and isinstance(value[0], Parameter)
            and isinstance(value[1], Parameter)
        ):
            self.real = value[0]
            self.imag = value[1]

        self._parameters = Parameters(name=name)
        self._parameters.extend([self.real, self.imag])

    def __repr__(self):
        return "SLD([{real!r}, {imag!r}]," " name={name!r})".format(
            **self.__dict__
        )

    def __complex__(self):
        sldc = complex(self.real.value, self.imag.value)
        return sldc

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters
        # p = Parameters(name=self.name)
        # p.extend([self.real, self.imag])
        # return p


class MaterialSLD(Scatterer):
    """
    Object representing SLD of a chemical formula.
    You can fit the mass density of the material.

    Parameters
    ----------
    formula : str
        Chemical formula
    density : float or Parameter
        mass density of compound in g / cm**3
    probe : {'x-ray', 'neutron'}, optional
        Are you using neutrons or X-rays?
    wavelength : float, optional
        wavelength of radiation (Angstrom)
    name : str, optional
        Name of material

    Notes
    -----
    You need to have the `periodictable` package installed to use this object.
    An SLD object can be used to create a Slab:

    >>> # a MaterialSLD object representing Silicon Dioxide
    >>> sio2 = MaterialSLD('SiO2', 2.2, name='SiO2')
    >>> # create a silica slab of SiO2 20 A in thickness, with a 3 A roughness
    >>> sio2_layer = sio2(20, 3)
    >>> # allow the mass density of the silica to vary between 2.1 and 2.3
    >>> # g/cm**3
    >>> sio2.density.setp(vary=True, bounds=(2.1, 2.3))
    """

    def __init__(
        self, formula, density, probe="neutron", wavelength=1.8, name=""
    ):
        import periodictable as pt

        super(MaterialSLD, self).__init__(name=name)

        self.__formula = pt.formula(formula)
        self._compound = formula
        self.density = possibly_create_parameter(density, name="density")
        if probe.lower() not in ["x-ray", "neutron"]:
            raise RuntimeError("'probe' must be one of 'x-ray' or 'neutron'")
        self.probe = probe.lower()
        self.wavelength = wavelength

        self._parameters = Parameters(name=name)
        self._parameters.extend([self.density])

    def __repr__(self):
        d = {
            "compound": self._compound,
            "density": self.density,
            "wavelength": self.wavelength,
            "probe": self.probe,
            "name": self.name,
        }
        return (
            "MaterialSLD({compound!r}, {density!r}, probe={probe!r},"
            " wavelength={wavelength!r}, name={name!r})".format(**d)
        )

    @property
    def formula(self):
        return self._compound

    @formula.setter
    def formula(self, formula):
        import periodictable as pt

        self.__formula = pt.formula(formula)
        self._compound = formula

    def __complex__(self):
        import periodictable as pt

        if self.probe == "neutron":
            sldc = pt.neutron_sld(
                self.__formula,
                density=self.density.value,
                wavelength=self.wavelength,
            )
        elif self.probe == "x-ray":
            sldc = pt.xray_sld(
                self.__formula,
                density=self.density.value,
                wavelength=self.wavelength,
            )
        return complex(sldc[0], sldc[1])

    @property
    def parameters(self):
        return self._parameters


class Component(object):
    """
    A base class for describing the structure of a subset of an interface.

    Parameters
    ----------
    name : str, optional
        The name associated with the Component

    Notes
    -----
    By setting the `Component.interfaces` property one can control the
    type of interfacial roughness between all the layers of an interfacial
    profile.
    """

    def __init__(self, name=""):
        self.name = name
        self._interfaces = None

    def __or__(self, other):
        """
        OR'ing components can create a :class:`Structure`.

        Parameters
        ----------
        other: refnx.reflect.Structure, refnx.reflect.Component
            Combines with this component to make a Structure

        Returns
        -------
        s: refnx.reflect.Structure
            The created Structure

        Examples
        --------

        >>> air = SLD(0, name='air')
        >>> sio2 = SLD(3.47, name='SiO2')
        >>> si = SLD(2.07, name='Si')
        >>> structure = air | sio2(20, 3) | si(0, 3)

        """
        # c = self | other
        p = Structure()
        p |= self
        p |= other
        return p

    def __mul__(self, n):
        """
        MUL'ing components makes them repeat.

        Parameters
        ----------
        n: int
            How many times you want to repeat the Component

        Returns
        -------
        s: refnx.reflect.Structure
            The created Structure
        """
        # convert to integer, should raise an error if there's a problem
        n = operator.index(n)
        if n < 1:
            return Structure()
        elif n == 1:
            return self
        else:
            s = Structure()
            s.extend([self] * n)
            return s

    def __str__(self):
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component
        """
        raise NotImplementedError(
            "A component should override the parameters " "property"
        )

    @property
    def interfaces(self):
        """
        The interfacial roughness type between each layer in `Component.slabs`.
        Should be one of {None, :class:`Interface`, or sequence of
        :class:`Interface`}.
        """
        return self._interfaces

    @interfaces.setter
    def interfaces(self, interfaces):
        # Sentinel for default roughness.
        if interfaces is None:
            self._interfaces = None
            return

        if isinstance(interfaces, Interface):
            self._interfaces = interfaces
            return

        # this will raise TypeError is interfaces is not iterable
        _interfaces = [i for i in interfaces if isinstance(i, Interface)]

        if len(_interfaces) == 1:
            self._interfaces = _interfaces[0]
            return

        n_slabs = len(self.slabs())
        if len(_interfaces) == n_slabs:
            self._interfaces = _interfaces
        else:
            raise ValueError(
                "Interface property must be set with one of:"
                " {None, Interface, sequence of Interface. If a"
                " sequence is provided it must have the same"
                " length as `Component.slabs`."
            )

    def slabs(self, structure=None):
        """
        The slab representation of this component

        Parameters
        ----------
        structure : refnx.reflect.Structure
            The Structure hosting the Component.

        Returns
        -------
        slabs : np.ndarray
            Slab representation of this Component.
            Has shape (N, 5).

            - slab[N, 0]
               thickness of layer N
            - slab[N, 1]
               SLD.real of layer N (not including solvent)
            - slab[N, 2]
               *overall* SLD.imag of layer N (not including solvent)
            - slab[N, 3]
               roughness between layer N and N-1
            - slab[N, 4]
               volume fraction of solvent in layer N.

        If a Component returns None, then it doesn't have any slabs.
        """

        raise NotImplementedError(
            "A component should override the slabs " "property"
        )

    def logp(self):
        """
        The log-probability that this Component adds to the total log-prior
        term. Do not include log-probability terms for the actual parameters,
        these are automatically included elsewhere.

        Returns
        -------
        logp : float
            Log-probability
        """
        return 0


class Slab(Component):
    """
    A slab component has uniform SLD over its thickness.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld : :class:`refnx.reflect.Scatterer`, complex, or float
        (complex) SLD of film (/1e-6 Angstrom**2)
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    vfsolv : refnx.analysis.Parameter or float
        Volume fraction of solvent [0, 1]
    interface : {:class:`Interface`, None}, optional
        The type of interfacial roughness associated with the Slab.
        If `None`, then the default interfacial roughness is an Error
        function (also known as Gaussian roughness).
    """

    def __init__(self, thick, sld, rough, name="", vfsolv=0, interface=None):
        super(Slab, self).__init__(name=name)
        self.thick = possibly_create_parameter(thick, name=f"{name} - thick")
        if isinstance(sld, Scatterer):
            self.sld = sld
        else:
            self.sld = SLD(sld)
        self.rough = possibly_create_parameter(rough, name=f"{name} - rough")
        self.vfsolv = possibly_create_parameter(
            vfsolv, name=f"{name} - volfrac solvent", bounds=(0.0, 1.0)
        )

        p = Parameters(name=self.name)
        p.extend([self.thick])
        p.extend(self.sld.parameters)
        p.extend([self.rough, self.vfsolv])

        self._parameters = p
        self.interfaces = interface

    def __repr__(self):
        return (
            f"Slab({self.thick!r}, {self.sld!r}, {self.rough!r},"
            f" name={self.name!r}, vfsolv={self.vfsolv!r},"
            f" interface={self.interfaces!r})"
        )

    def __str__(self):
        # sld = repr(self.sld)
        #
        # s = 'Slab: {0}\n    thick = {1} Å, {2}, rough = {3} Å,
        #      \u03D5_solv = {4}'
        # t = s.format(self.name, self.thick.value, sld, self.rough.value,
        #              self.vfsolv.value)
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """
        sldc = complex(self.sld)
        return np.array(
            [
                [
                    self.thick.value,
                    sldc.real,
                    sldc.imag,
                    self.rough.value,
                    self.vfsolv.value,
                ]
            ]
        )


class MixedSlab(Component):
    """
    A slab component made of several components

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld_list : sequence of {refnx.reflect.Scatterer, complex, float}
        Sequence of (complex) SLDs that are contained in film
        (/1e-6 Angstrom**2)
    vf_list : sequence of refnx.analysis.Parameter or float
        relative volume fractions of each of the materials contained in the
        film.
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    vfsolv : refnx.analysis.Parameter or float
        Volume fraction of solvent [0, 1]
    interface : {:class:`Interface`, None}, optional
        The type of interfacial roughness associated with the Slab.
        If `None`, then the default interfacial roughness is an Error
        function (also known as Gaussian roughness).

    Notes
    -----
    The SLD of this Slab is calculated using the normalised volume fractions of
    each of the constituent Scatterers:

    >>> np.sum([complex(sld) * vf / np.sum(vf_list) for sld, vf in
    ...         zip(sld_list, vf_list)]).

    The overall SLD then takes into account the volume fraction of solvent,
    `vfsolv`.
    """

    def __init__(
        self,
        thick,
        sld_list,
        vf_list,
        rough,
        name="",
        vfsolv=0,
        interface=None,
    ):
        super(MixedSlab, self).__init__(name=name)
        self.thick = possibly_create_parameter(thick, name="%s - thick" % name)

        self.sld = []
        self.vf = []
        self._sld_parameters = Parameters(name=f"{name} - slds")
        self._vf_parameters = Parameters(name=f"{name} - volfracs")

        i = 0
        for s, v in zip(sld_list, vf_list):
            if isinstance(s, Scatterer):
                self.sld.append(s)
            else:
                self.sld.append(SLD(s))

            self._sld_parameters.append(self.sld[-1].parameters)

            vf = possibly_create_parameter(
                v, name=f"vf{i} - {name}", bounds=(0.0, 1.0)
            )
            self.vf.append(vf)
            self._vf_parameters.append(vf)
            i += 1

        self.vfsolv = possibly_create_parameter(
            vfsolv, name=f"{name} - volfrac solvent", bounds=(0.0, 1.0)
        )
        self.rough = possibly_create_parameter(rough, name=f"{name} - rough")

        p = Parameters(name=self.name)
        p.append(self.thick)
        p.extend(self._sld_parameters)
        p.extend(self._vf_parameters)
        p.extend([self.vfsolv, self.rough])

        self._parameters = p
        self.interfaces = interface

    def __repr__(self):
        return (
            f"MixedSlab({self.thick!r}, {self.sld!r}, {self.vf!r},"
            f" {self.rough!r}, vfsolv={self.vfsolv!r}, name={self.name!r},"
            f" interface={self.interfaces!r})"
        )

    def __str__(self):
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """
        vfs = np.array(self._vf_parameters)
        sum_vfs = np.sum(vfs)

        sldc = np.sum(
            [complex(sld) * vf / sum_vfs for sld, vf in zip(self.sld, vfs)]
        )

        return np.array(
            [
                [
                    self.thick.value,
                    sldc.real,
                    sldc.imag,
                    self.rough.value,
                    self.vfsolv.value,
                ]
            ]
        )


class Stack(Component, UserList):
    r"""
    A series of Components to be considered as one. When part of a Structure
    the Stack can represent a multilayer by setting the `repeats` attribute.

    Parameters
    ----------
    components : sequence
        A series of Components to initialise the stack with
    name : str
        Name of the Stack
    repeats : number, Parameter
        When viewed from a parent Structure the Components in this Stack will
        appear to be repeated `repeats` times. Internally `repeats` is rounded
        to the nearest integer before use, allowing it to be used as a fitting
        parameter.

    Notes
    -----
    To add Components to the Stack you can:

        - initialise the object with a list of Components
        - utilise list methods (`extend`, `append`, `insert`, etc)
        - Add by `__ior__`  (e.g. `stack |= component`)

    You can't use `__or__` to add Components to a stack (e.g.
    ``Stack() | component``) ORing a Stack with other Components will make a
    Structure.
    """

    def __init__(self, components=(), name="", repeats=1):
        Component.__init__(self, name=name)
        UserList.__init__(self)  # explicit calls without super

        self.repeats = possibly_create_parameter(repeats, "repeats")
        self.repeats.bounds.lb = 1

        # if you provide a list of components to start with, then initialise
        # the Stack from that
        for c in components:
            if isinstance(c, Component):
                self.data.append(c)
            else:
                raise ValueError(
                    "You can only initialise a Stack with" " Components"
                )

    def __setitem__(self, i, v):
        self.data[i] = v

    def __str__(self):
        s = list()
        s.append("{:=>80}".format(""))

        s.append(
            "Stack start: {} repeats".format(round(abs(self.repeats.value)))
        )
        for component in self:
            s.append(str(component))
        s.append("Stack finish")
        s.append("{:=>80}".format(""))

        return "\n".join(s)

    def __repr__(self):
        return (
            "Stack(name={name!r},"
            " components={data!r},"
            " repeats={repeats!r})".format(**self.__dict__)
        )

    def append(self, item):
        """
        Append a :class:`Component` to the Stack.

        Parameters
        ----------
        item: refnx.reflect.Component
            The component to be added.
        """
        if isinstance(item, Scatterer):
            self.append(item())
            return

        if not isinstance(item, Component):
            raise ValueError(
                "You can only add Component objects to a" " structure"
            )
        self.data.append(item)

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`

        Notes
        -----
        The overall set of slabs returned by this method consists of the
        concatenated constituent Component slabs repeated `Stack.repeats`
        times.

        """
        if not len(self):
            return None

        # a sub stack member may want to know what the solvent is.
        if structure is not None:
            self.solvent = structure.solvent

        repeats = round(abs(self.repeats.value))

        slabs = np.concatenate(
            [c.slabs(structure=self) for c in self.components]
        )

        if repeats > 1:
            slabs = np.concatenate([slabs] * repeats)

        if hasattr(self, "solvent"):
            delattr(self, "solvent")

        return slabs

    def _interfaces_get(self):
        repeats = round(abs(self.repeats.value))
        interfaces = list(flatten([i.interfaces for i in self.data]))

        if repeats > 1:
            interfaces = interfaces * repeats

        return interfaces

    def _interfaces_set(self, interfaces):
        raise RuntimeError(
            "Cannot set interfaces property for a Stack"
            " Component. Please set the interfaces property"
            " for the constituent Components."
        )

    # override the interfaces property for this subclass
    interfaces = property(_interfaces_get, _interfaces_set)

    @property
    def components(self):
        """
        The list of components in the sample.
        """
        return self.data

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters`, all the parameters associated with
        this structure.

        """
        p = Parameters(name="Stack - {0}".format(self.name))
        p.append(self.repeats)
        p.extend([component.parameters for component in self.components])
        return p

    def __ior__(self, other):
        """
        Build a Stack by `IOR`'ing.

        Parameters
        ----------
        other: :class:`Component`, :class:`Scatterer`
            The object to add to the structure.

        """
        # self |= other
        if isinstance(other, Component):
            self.append(other)
        elif isinstance(other, Scatterer):
            slab = other(0, 0)
            self.append(slab)
        else:
            raise ValueError()
        return self


def _profile_slicer(z, sld_profile, slice_size=None):
    """
    Converts a scattering length density profile into a Structure by
    approximating with Slabs.

    Parameters
    ----------
    z : array-like
        Distance (Angstrom) through the interface at which the SLD profile is
        given.
    sld_profile : array-like
        Scattering length density (10**-6 Angstrom**-2) at a given distance
        through the interface. Both the real and imaginary terms of the SLD can
        be provided - either by making `sld_profile` a complex array, by
        supplying an array with two columns (representing the real and
        imaginary parts).
    slice_size : None, float, optional
        if `slice_size is None` then `np.min(np.diff(z))/4` is used to
        determine the rough size of the created slabs. Otherwise
        `float(slice_size)` is used.

    Returns
    -------
    structure : Structure
        A Structure representation of the sld profile

    Notes
    -----
    `sld_profile` is quadratically interpolated to obtain equally spaced
    points. In testing the round trip structure->sld_profile->structure the
    maximum relative difference in reflectivity profiles from the original and
    final structures is on the order of fractions of a percent, with the
    largest difference around the critical edge.
    """
    sld = np.asfarray(sld_profile, dtype=complex)
    if len(sld.shape) > 1 and sld.shape[1] == 2:
        sld[:, 0].imag = sld[:, 1].real
        sld = sld[:, 0]

    real_interp = interp1d(z, sld.real, kind="quadratic")
    imag_interp = interp1d(z, sld.imag, kind="quadratic")

    if slice_size is None:
        slice_size = np.min(np.diff(z)) / 4
    else:
        slice_size = float(slice_size)

    # figure out the z values to calculate the slabs at
    z_min, z_max = np.min(z), np.max(z)
    n_steps = np.ceil((z_max - z_min) / slice_size)
    zeds = np.linspace(z_min, z_max, int(n_steps) + 1)

    # this is the true thickness of the slab
    slice_size = np.diff(zeds)[0]
    zeds -= slice_size / 2
    zeds = zeds[1:]

    reals = real_interp(zeds)
    imags = imag_interp(zeds)

    slabs = [
        Slab(slice_size, complex(real, imag), 0)
        for real, imag in zip(reals, imags)
    ]
    structure = Structure(name="sliced sld profile")
    structure.extend(slabs)

    return structure


def sld_profile(slabs, z=None):
    """
    Calculates an SLD profile, as a function of distance through the
    interface.

    Parameters
    ----------
    z : float
        Interfacial distance (Angstrom) measured from interface between the
        fronting medium and the first layer.

    Returns
    -------
    sld : float
        Scattering length density / 1e-6 Angstrom**-2

    Notes
    -----
    This can be called in vectorised fashion.
    """
    nlayers = np.size(slabs, 0) - 2

    # work on a copy of the input array
    layers = np.copy(slabs)
    layers[:, 0] = np.fabs(slabs[:, 0])
    layers[:, 3] = np.fabs(slabs[:, 3])
    # bounding layers should have zero thickness
    layers[0, 0] = layers[-1, 0] = 0

    # distance of each interface from the fronting interface
    dist = np.cumsum(layers[:-1, 0])

    # workout how much space the SLD profile should encompass
    # (if z array not provided)
    if z is None:
        zstart = -5 - 4 * np.fabs(slabs[1, 3])
        zend = 5 + dist[-1] + 4 * layers[-1, 3]
        zed = np.linspace(zstart, zend, num=500)
    else:
        zed = np.asfarray(z)

    # the output array
    sld = np.ones_like(zed, dtype=float) * layers[0, 1]

    # work out the step in SLD at an interface
    delta_rho = layers[1:, 1] - layers[:-1, 1]

    # use erf for roughness function, but step if the roughness is zero
    step_f = Step()
    erf_f = Erf()
    sigma = layers[1:, 3]

    # accumulate the SLD of each step.
    for i in range(nlayers + 1):
        f = erf_f
        if sigma[i] == 0:
            f = step_f
        sld += delta_rho[i] * f(zed, scale=sigma[i], loc=dist[i])

    return zed, sld
