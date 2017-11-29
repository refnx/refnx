# -*- coding: utf-8 -*-

from __future__ import division
from six.moves import UserList

import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d

try:
    from refnx.reflect import _creflect as refcalc
except ImportError:
    print('WARNING, Using slow reflectivity calculation')
    from refnx.reflect import _reflect as refcalc
from refnx.analysis import Parameters, Parameter, possibly_create_parameter


class Structure(UserList):
    """
    Represents the interfacial Structure of a reflectometry sample.
    Successive Components are added to the Structure to construct the
    interface.

    Parameters
    ----------
    name : str
        Name of this structure
    solvent : str
        Specifies whether the 'backing' or 'fronting' semi-infinite medium
        is used to solvate components. You would typically use 'backing'
        for neutron reflectometry, with solvation by the material in
        Structure[-1]. X-ray reflectometry would typically be solvated by
        the 'fronting' material in Structure[0].
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
    order is reversed. The slab order is reversed before the solvation
    calculation is done. I.e. if `Structure.solvent == 'backing'` and
    `Structure.reverse_structure is True` then the material that solvates
    the system is the component in `Structure[0]`, which corresponds to
    `Structure.slab[-1]`.
    The profile contraction specified by the `contract` keyword can improve
    calculation time for Structures created with microslicing (such as
    analytical profiles). If you use this option it is recommended to check
    the reflectivity signal with and without contraction to ensure they are
    comparable.

    """
    def __init__(self, name='', solvent='backing', reverse_structure=False,
                 contract=0):
        super(Structure, self).__init__()
        self._name = name
        if solvent not in ['backing', 'fronting']:
            raise ValueError("solvent must either be the fronting or backing"
                             " medium")

        #: **str** specifies whether `backing` or `fronting` semi-infinite
        #: medium is used to solvate components.
        self.solvent = solvent
        self._reverse_structure = bool(reverse_structure)
        #: **float** if contract > 0 then an attempt to contract/shrink the
        #: slab representation is made. Use larger values for coarser profiles
        #: (and vice versa). A typical starting value to try might be 1.0.
        self.contract = contract
        # self._parameters = Parameters(name=name)

    def __copy__(self):
        s = Structure(self.name, solvent=self.solvent)
        s.data = self.data.copy()
        return s

    def __setitem__(self, i, v):
        self.data[i] = v
        # self.update()

    def append(self, item):
        if isinstance(item, SLD):
            self.append(item())
            return

        if not isinstance(item, Component):
            raise ValueError("You can only add Component objects to a"
                             " structure")
        super(Structure, self).append(item)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

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

    @property
    def slabs(self):
        r"""
        :class:`np.ndarray` - slab representation of this structure.
        Has shape (N, 5).

        - slab[N, 0]
           thickness of layer N
        - slab[N, 1]
           overall SLD.real of layer N (material AND solvent)
        - slab[N, 2]
           overall SLD.imag of layer N (material AND solvent)
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

        # could possibly speed up by allocating a larger array, filling,
        # then trimming
        growth_size = 100
        slabs = np.zeros((growth_size, 5))
        i = 0
        for component in self.components:
            additional_slabs = component.slabs
            new_slabs = len(additional_slabs)

            if new_slabs > len(slabs) - i:
                new_rows = len(slabs) + max(growth_size, new_slabs)
                slabs = np.resize(slabs, (new_rows, 5))
                slabs[i:] = 0

            slabs[i:i + new_slabs] = additional_slabs
            i += new_slabs

        slabs = slabs[:i]

        # if the slab representation needs to be reversed.
        if self.reverse_structure:
            roughnesses = slabs[1:, 3]
            slabs = np.flipud(slabs)
            slabs[1:, 3] = roughnesses[::-1]
            slabs[0, 3] = 0.

        if len(self) > 2:
            if self.solvent == 'backing':
                solvent_slab = slabs[-1]
            if self.solvent == 'fronting':
                solvent_slab = slabs[0]

            # overall SLD is a weighted average
            slabs[1:-1] = self.overall_sld(slabs[1:-1], solvent_slab)

        if self.contract > 0:
            return _contract_by_area(slabs, self.contract)
        else:
            return slabs

    @staticmethod
    def overall_sld(slabs, solvent_slab):
        """
        Performs a volume fraction weighted average of the material SLD in a
        layer and the solvent in a layer.

        Parameters
        ----------
        slabs : np.ndarray
            Slab representation of the layers to be averaged.
        solvent_slab: np.ndarray
            Slab representation of the solvent layer

        Returns
        -------
        averaged_slabs : np.ndarray
            the averaged slabs.
        """
        slabs[..., 1] = slabs[..., 1] * (1 - slabs[..., 4])
        slabs[..., 2] = slabs[..., 2] * (1 - slabs[..., 4])
        slabs[..., 1] += solvent_slab[..., 1] * slabs[..., 4]
        slabs[..., 2] += solvent_slab[..., 2] * slabs[..., 4]
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
        """
        return refcalc.abeles(q, self.slabs[..., :4], threads=threads)

    def sld_profile(self, z=None):
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
            Scattering length density / 1e-6 $\AA^-2$

        Notes
        -----
        This can be called in vectorised fashion.
        """
        slabs = self.slabs
        if ((slabs is None) or
                (len(slabs) < 2) or
                (not isinstance(self.data[0], Slab)) or
                (not isinstance(self.data[-1], Slab))):
            raise ValueError("Structure requires fronting and backing"
                             " Slabs in order to calculate.")

        nlayers = np.size(slabs, 0) - 2

        if z is None:
            if not nlayers:
                zstart = -5 - 4 * np.fabs(slabs[-1, 3])
                zend = 5 + 4 * np.fabs(slabs[-1, 3])
            else:
                zstart = -5 - 4 * np.fabs(slabs[1, 3])
                sum_thick = np.sum(np.fabs(slabs[1:-1, 0]))
                zend = 5 + sum_thick + 4 * np.fabs(slabs[-1, 3])

            z = np.linspace(zstart, zend, num=500)

        def sld_z(zed):
            sld = np.zeros_like(zed)

            dist = 0
            thick = 0
            for ii in range(nlayers + 1):
                if ii == 0:
                    if nlayers:
                        deltarho = -slabs[0, 1] + slabs[1, 1]
                        thick = 0
                        sigma = np.fabs(slabs[1, 3])
                    else:
                        sigma = np.fabs(slabs[-1, 3])
                        deltarho = -slabs[0, 1] + slabs[-1, 1]
                elif ii == nlayers:
                    sld1 = slabs[ii, 1]
                    deltarho = -sld1 + slabs[-1, 1]
                    thick = np.fabs(slabs[ii, 0])
                    sigma = np.fabs(slabs[-1, 3])
                else:
                    sld1 = slabs[ii, 1]
                    sld2 = slabs[ii + 1, 1]
                    deltarho = -sld1 + sld2
                    thick = np.fabs(slabs[ii, 0])
                    sigma = np.fabs(slabs[ii + 1, 3])

                dist += thick

                # if sigma=0 then the computer goes haywire (division by zero),
                # so say it's vanishingly small
                if sigma == 0:
                    sigma += 1e-3

                # summ += deltarho * (norm.cdf((zed - dist)/sigma))
                sld += (deltarho *
                        (0.5 +
                         0.5 *
                         erf((zed - dist) / (sigma * np.sqrt(2.)))))

            return sld

        return z, sld_z(z) + slabs[0, 1]

    def __ior__(self, other):
        # self |= other
        if isinstance(other, Component):
            self.append(other)
        elif isinstance(other, Structure):
            self.extend(other.data)
        elif isinstance(other, SLD):
            slab = other(0, 0)
            self.append(slab)
        else:
            raise ValueError()

        return self

    def __or__(self, other):
        # c = self | other
        p = Structure()
        p |= self
        p |= other
        return p

    @property
    def components(self):
        return self.data

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters`, all the parameters associated with
        this structure.

        """
        p = Parameters(name='Structure - {0}'.format(self.name))
        p.extend([component.parameters for component in self.components])
        return p

    def lnprob(self):
        """
        log-probability for the interfacial structure. Note that if a given
        component is present more than once in a Structure then it's log-prob
        will be counted twice.

        Returns
        -------
        lnprob : float
            log-prior for the Structure.
        """
        lnprob = 0
        for component in self.components:
            lnprob += component.lnprob()

        return lnprob


class SLD(object):
    """
    Object representing freely varying SLD of a material

    Parameters
    ----------
    value : float or complex
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
    >>> sio2_layer = SLD(20, 3)

    """
    def __init__(self, value, name=''):
        self.name = name
        if isinstance(value, complex):
            self.real = Parameter(value.real, name='%s - sld' % name)
            self.imag = Parameter(value.imag, name='%s - isld' % name)
        elif isinstance(value, SLD):
            self.real = value.real
            self.imag = value.imag
        else:
            self.real = Parameter(value, name='%s - sld' % name)
            self.imag = Parameter(0, name='%s - isld' % name)

        self._parameters = Parameters(name=name)
        self._parameters.extend([self.real, self.imag])

    def __call__(self, thick=0, rough=0):
        return Slab(thick, self, rough, name=self.name)

    def __or__(self, other):
        # c = self | other
        slab = self()
        return slab | other

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


class Component(object):

    def __init__(self):
        pass

    def __or__(self, other):
        # c = self | other
        p = Structure()
        p |= self
        p |= other
        return p

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        raise NotImplementedError("A component should override the parameters "
                                  "property")

    @property
    def slabs(self):
        """
        The slab representation of this component

        """

        raise NotImplementedError("A component should override the slabs "
                                  "property")

    def lnprob(self):
        return 0


class Slab(Component):
    """
    A slab component has uniform SLD over its thickness.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld : refnx.reflect.SLD, complex, or float
        (complex) SLD of film (/1e-6 Angstrom**2)
    rough : float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    vfsolv : refnx.analysis.Parameter or float
        Volume fraction of solvent [0, 1]

    """

    def __init__(self, thick, sld, rough, name='', vfsolv=0):
        super(Slab, self).__init__()
        self.thick = possibly_create_parameter(thick,
                                               name='%s - thick' % name)
        if isinstance(sld, SLD):
            self.sld = sld
        else:
            self.sld = SLD(sld)
        self.rough = possibly_create_parameter(rough,
                                               name='%s - rough' % name)
        self.vfsolv = (
            possibly_create_parameter(vfsolv,
                                      name='%s - volfrac solvent' % name))
        self.name = name

        p = Parameters(name=self.name)
        p.extend([self.thick, self.sld.real, self.sld.imag,
                  self.rough, self.vfsolv])

        self._parameters = p

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    @property
    def slabs(self):
        """
        slab representation of this component. See :class:`Structure.slabs`
        """
        return np.atleast_2d(np.array([self.thick.value,
                                       self.sld.real.value,
                                       self.sld.imag.value,
                                       self.rough.value,
                                       self.vfsolv.value]))


class CompositeComponent(Component):
    """
    A series of components to be considered as one.
    """
    def __init__(self, components):
        pass


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
        if `slice_size is None` then `np.min(np.ediff1d(z))/4` is used to
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

    real_interp = interp1d(z, sld.real, kind='quadratic')
    imag_interp = interp1d(z, sld.imag, kind='quadratic')

    if slice_size is None:
        slice_size = np.min(np.ediff1d(z)) / 4
    else:
        slice_size = float(slice_size)

    # figure out the z values to calculate the slabs at
    z_min, z_max = np.min(z), np.max(z)
    n_steps = np.ceil((z_max - z_min) / slice_size)
    zeds = np.linspace(z_min, z_max, int(n_steps) + 1)

    # this is the true thickness of the slab
    slice_size = np.ediff1d(zeds)[0]
    zeds -= slice_size / 2
    zeds = zeds[1:]

    reals = real_interp(zeds)
    imags = imag_interp(zeds)

    slabs = [Slab(slice_size, complex(real, imag), 0) for
             real, imag in zip(reals, imags)]
    structure = Structure(name='sliced sld profile')
    structure.extend(slabs)

    return structure


# The following slab contraction code was translated from C code in
# the refl1d project.
def _contract_by_area(slabs, dA=0.5):
    """
    Shrinks a slab representation to a reduced number of layers. This can
    reduced calculation times.

    Parameters
    ----------
    slabs : array
        Has shape (N, 5).

            slab[N, 0] - thickness of layer N
            slab[N, 1] - overall SLD.real of layer N (material AND solvent)
            slab[N, 2] - overall SLD.imag of layer N (material AND solvent)
            slab[N, 3] - roughness between layer N and N-1
            slab[N, 4] - volume fraction of solvent in layer N.
                         (1 - solvent_volfrac = material_volfrac)

    dA : float
        Larger values coarsen the profile to a greater extent, and vice versa.

    Returns
    -------
    contract_slab : array
        Contracted slab representation.

    Notes
    -----
    The reflectivity profiles from both contracted and uncontracted profiles
    should be compared to check for accuracy.
    """

    # In refl1d the first slab is the substrate, the orded is reversed here.
    # In the following code the slabs are traversed from the backing towards
    # the fronting.
    newslabs = np.copy(slabs)[::-1]
    d = newslabs[:, 0]
    rho = newslabs[:, 1]
    irho = newslabs[:, 2]
    sigma = newslabs[:, 3]
    vfsolv = newslabs[:, 4]

    n = np.size(d, 0)
    i = newi = 1  # Skip the substrate

    while i < n:
        # Get ready for the next layer
        # Accumulation of the first row happens in the inner loop
        dz = rhoarea = irhoarea = vfsolvarea = 0.
        rholo = rhohi = rho[i]
        irholo = irhohi = irho[i]

        # Accumulate slices into layer
        while True:
            # Accumulate next slice
            dz += d[i]
            rhoarea += d[i] * rho[i]
            irhoarea += d[i] * irho[i]
            vfsolvarea += d[i] * vfsolv[i]

            i += 1
            # If no more slices or sigma != 0, break immediately
            if i == n or sigma[i - 1] != 0.:
                break

            # If next slice won't fit, break
            if rho[i] < rholo:
                rholo = rho[i]
            if rho[i] > rhohi:
                rhohi = rho[i]
            if (rhohi - rholo) * (dz + d[i]) > dA:
                break

            if irho[i] < irholo:
                irholo = irho[i]
            if irho[i] > irhohi:
                irhohi = irho[i]
            if (irhohi - irholo) * (dz + d[i]) > dA:
                break

        # Save the layer
        d[newi] = dz
        if i == n:
            # printf("contract: adding final sld at %d\n",newi)
            # Last layer uses surface values
            rho[newi] = rho[n - 1]
            irho[newi] = irho[n - 1]
            vfsolv[newi] = vfsolv[n - 1]
        else:
            # Middle layers uses average values
            rho[newi] = rhoarea / dz
            irho[newi] = irhoarea / dz
            sigma[newi] = sigma[i - 1]
            vfsolv[newi] = vfsolvarea / dz
        # First layer uses substrate values
        newi += 1

    return newslabs[:newi][::-1]
