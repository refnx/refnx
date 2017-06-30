from __future__ import division
from six.moves import UserList

import numpy as np
from scipy.special import erf

try:
    from refnx.reflect import _creflect as refcalc
except ImportError:
    print('WARNING, Using slow reflectivity calculation')
    from refnx.reflect import _reflect as refcalc
from refnx.analysis import Parameters, Parameter, possibly_create_parameter


class Structure(UserList):
    def __init__(self, name='', solvent='backing'):
        super(Structure, self).__init__()
        self._name = name
        if solvent not in ['backing', 'fronting']:
            raise ValueError("solvent must either be the fronting or backing"
                             " medium")

        self.solvent = solvent
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
    def slabs(self):
        """
        Slab representation of this structure

        Returns
        -------
        slabs : np.ndarray
            Has shape (N, 5).
            slab[N, 0] - thickness of layer N
            slab[N, 1] - overall SLD.real of layer N (material AND solvent)
            slab[N, 2] - overall SLD.imag of layer N (material AND solvent)
            slab[N, 3] - roughness between layer N and N-1
            slab[N, 4] - volume fraction of solvent in layer N.
                         (1 - solvent_volfrac = material_volfrac)
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
                slabs = np.resize(slabs, (len(slabs) + growth_size, 5))
                slabs[i:] = 0

            slabs[i:i + new_slabs] = additional_slabs
            i += new_slabs

        slabs = slabs[:i]

        if len(self) > 2:
            if self.solvent == 'backing':
                solvent_real = slabs[-1, 1]
                solvent_imag = slabs[-1, 2]
            if self.solvent == 'fronting':
                solvent_real = slabs[0, 1]
                solvent_imag = slabs[0, 2]

            # overall SLD is a weighted average
            slabs[1:-1, 1] = slabs[1:-1, 1] * (1 - slabs[1:-1, 4])
            slabs[1:-1, 2] = slabs[1:-1, 2] * (1 - slabs[1:-1, 4])
            slabs[1:-1, 1] += solvent_real * slabs[1:-1, 4]
            slabs[1:-1, 2] += solvent_imag * slabs[1:-1, 4]

        return slabs

    def reflectivity(self, q, workers=0):
        """
        Calculate theoretical reflectivity of this structure

        Parameters
        ----------
        q : array-like
            Q values for evaluation
        workers : int, optional
            Specifies the number of threads for parallel calculation. This
            option is only applicable if you are using the ``_creflect``
            module. The option is ignored if using the pure python calculator,
            ``_reflect``. If `workers == 0` then all available processors are
            used.
        """
        return refcalc.abeles(q, self.slabs[..., :4], workers=workers)

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
        # return self._parameters
        p = Parameters(name=self.name)
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
    """
    def __init__(self, value, name=''):
        """
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
        >>> # create a Slab of SiO2 20 A in thickness, with a 3 A roughness
        >>> sio2_layer = SLD(20, 3)
        """
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
        raise NotImplementedError("A component should override the parameters "
                                  "property")

    @property
    def slabs(self):
        raise NotImplementedError("A component should override the slabs "
                                  "property")

    def lnprob(self):
        return 0


class Slab(Component):
    """
    A slab component has uniform SLD over its thickness.
    """

    def __init__(self, thick, sld, rough, name='', vfsolv=0):
        """
        Parameters
        ----------
        thick : Parameter or float
            thickness of slab (Angstrom)
        sld : SLD instance, complex, or float
            (complex) SLD of film (/1e-6 Angstrom**2)
        rough : float
            roughness on top of this slab (Angstrom)
        name : str
            Name of this slab
        vfsolv : Parameter or float
            Volume fraction of solvent [0, 1]
        """
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
        return self._parameters

    @property
    def slabs(self):
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
