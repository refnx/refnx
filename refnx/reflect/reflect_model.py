import abc
import math
import numbers
import warnings

import numpy as np
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline

try:
    from refnx.reflect import _creflect as refcalc
except ImportError:
    print('WARNING, Using slow reflectivity calculation')
    from refnx.reflect import _reflect as refcalc
from refnx.analysis import (Parameters, Parameter, possibly_create_parameter)


# some definitions for resolution smearing
_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5


class ReflectModel(object):
    r"""
    Parameters
    ----------
    structure : refnx.reflect.Structure
        The interfacial structure.
    scale : float or refnx.analysis.Parameter, optional
        scale factor. All model values are multiplied by this value before
        the background is added. This is turned into a Parameter during the
        construction of this object.
    bkg : float or refnx.analysis.Parameter, optional
        Q-independent constant background added to all model values. This is
        turned into a Parameter during the construction of this object.
    name : str, optional
        Name of the Model
    dq : float or refnx.analysis.Parameter, optional

        - `dq == 0` then no resolution smearing is employed.
        - `dq` is a float or refnx.analysis.Parameter
           a constant dQ/Q resolution smearing is employed.  For 5% resolution
           smearing supply 5.

        However, if `x_err` is supplied to the `model` method, then that
        overrides any setting given here. This value is turned into
        a Parameter during the construction of this object.
    threads: int, optional
        Specifies the number of threads for parallel calculation. This
        option is only applicable if you are using the ``_creflect``
        module. The option is ignored if using the pure python calculator,
        ``_reflect``. If `threads == -1` then all available processors are
        used.
    quad_order: int, optional
        the order of the Gaussian quadrature polynomial for doing the
        resolution smearing. default = 17. Don't choose less than 13. If
        quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
        quadrature will always work, but takes a _long_ time (2 or 3 orders
        of magnitude longer). Fixed quadrature will always take a lot less
        time. BUT it won't necessarily work across all samples. For
        example, 13 points may be fine for a thin layer, but will be
        atrocious at describing a multilayer with bragg peaks.

    """
    def __init__(self, structure, scale=1, bkg=1e-7, name='', dq=5.,
                 threads=-1, quad_order=17):
        self.name = name
        self._parameters = None
        self.threads = threads
        self.quad_order = quad_order

        # to make it more like a refnx.analysis.Model
        self.fitfunc = None

        # all reflectometry models need a scale factor and background
        self._scale = possibly_create_parameter(scale, name='scale')
        self._bkg = possibly_create_parameter(bkg, name='bkg')

        # we can optimize the resolution (but this is always overridden by
        # x_err if supplied. There is therefore possibly no dependence on it.
        self._dq = possibly_create_parameter(dq, name='dq - resolution')

        self._structure = None
        self.structure = structure

    def __call__(self, x, p=None, x_err=None):
        r"""
        Calculate the generative model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity
        """
        return self.model(x, p=p, x_err=x_err)

    def __repr__(self):
        return ("ReflectModel({_structure!r}, name={name!r},"
                " scale={_scale!r}, bkg={_bkg!r},"
                " dq={_dq!r}, threads={threads},"
                " quad_order={quad_order})".format(**self.__dict__))

    @property
    def dq(self):
        r"""
        :class:`refnx.analysis.Parameter`

            - `dq.value == 0`
               no resolution smearing is employed.
            - `dq.value > 0`
               a constant dQ/Q resolution smearing is employed.  For 5%
               resolution smearing supply 5. However, if `x_err` is supplied to
               the `model` method, then that overrides any setting reported
               here.

        """
        return self._dq

    @dq.setter
    def dq(self, value):
        self._dq.value = value

    @property
    def scale(self):
        r"""
        :class:`refnx.analysis.Parameter` - all model values are multiplied by
        this value before the background is added.

        """
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale.value = value

    @property
    def bkg(self):
        r"""
        :class:`refnx.analysis.Parameter` - linear background added to all
        model values.

        """
        return self._bkg

    @bkg.setter
    def bkg(self, value):
        self._bkg.value = value

    def model(self, x, p=None, x_err=None):
        r"""
        Calculate the reflectivity of this model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity

        """
        if p is not None:
            self.parameters.pvals = np.array(p)
        if x_err is None:
            # fallback to what this object was constructed with
            x_err = float(self.dq)

        return reflectivity(x, self.structure.slabs()[..., :4],
                            scale=self.scale.value,
                            bkg=self.bkg.value,
                            dq=x_err,
                            threads=self.threads,
                            quad_order=self.quad_order)

    def logp(self):
        r"""
        Additional log-probability terms for the reflectivity model. Do not
        include log-probability terms for model parameters, these are
        automatically included elsewhere.

        Returns
        -------
        logp : float
            log-probability of structure.

        """
        return self.structure.logp()

    @property
    def structure(self):
        r"""
        :class:`refnx.reflect.Structure` - object describing the interface of
        a reflectometry sample.

        """
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure
        p = Parameters(name='instrument parameters')
        p.extend([self.scale, self.bkg, self.dq])

        self._parameters = Parameters(name=self.name)
        self._parameters.extend([p, structure.parameters])

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters` - parameters associated with this
        model.

        """
        self.structure = self._structure
        return self._parameters


def reflectivity(q, slabs, scale=1., bkg=0., dq=5., quad_order=17,
                 threads=-1):
    r"""
    Abeles matrix formalism for calculating reflectivity from a stratified
    medium.

    Parameters
    ----------
    q : np.ndarray
        The qvalues required for the calculation.
        :math:`Q=\frac{4Pi}{\lambda}\sin(\Omega)`.
        Units = Angstrom**-1
    slabs : np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers

        - slabs[0, 0]
           ignored
        - slabs[N, 0]
           thickness of layer N
        - slabs[N+1, 0]
           ignored

        - slabs[0, 1]
           SLD_real of fronting (/1e-6 Angstrom**-2)
        - slabs[N, 1]
           SLD_real of layer N (/1e-6 Angstrom**-2)
        - slabs[-1, 1]
           SLD_real of backing (/1e-6 Angstrom**-2)

        - slabs[0, 2]
           SLD_imag of fronting (/1e-6 Angstrom**-2)
        - slabs[N, 2]
           iSLD_imag of layer N (/1e-6 Angstrom**-2)
        - slabs[-1, 2]
           iSLD_imag of backing (/1e-6 Angstrom**-2)

        - slabs[0, 3]
           ignored
        - slabs[N, 3]
           roughness between layer N-1/N
        - slabs[-1, 3]
           roughness between backing and layer N

    scale : float
        scale factor. All model values are multiplied by this value before
        the background is added
    bkg : float
        Q-independent constant background added to all model values.
    dq : float or np.ndarray, optional
        - `dq == 0`
           no resolution smearing is employed.
        - `dq` is a float
           a constant dQ/Q resolution smearing is employed.  For 5% resolution
           smearing supply 5.
        - `dq` is the same shape as q
           the array contains the FWHM of a Gaussian approximated resolution
           kernel. Point by point resolution smearing is employed.  Use this
           option if dQ/Q varies across your dataset.
        - `dq.ndim == q.ndim + 2` and `q.shape == dq[..., -3].shape`
           an individual resolution kernel is applied to each measurement
           point. This resolution kernel is a probability distribution function
           (PDF). `dqvals` will have the shape (qvals.shape, 2, M).  There are
           `M` points in the kernel. `dq[:, 0, :]` holds the q values for the
           kernel, `dq[:, 1, :]` gives the corresponding probability.
    quad_order: int, optional
        the order of the Gaussian quadrature polynomial for doing the
        resolution smearing. default = 17. Don't choose less than 13. If
        quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
        quadrature will always work, but takes a _long_ time (2 or 3 orders
        of magnitude longer). Fixed quadrature will always take a lot less
        time. BUT it won't necessarily work across all samples. For
        example, 13 points may be fine for a thin layer, but will be
        atrocious at describing a multilayer with bragg peaks.
    threads: int, optional
        Specifies the number of threads for parallel calculation. This
        option is only applicable if you are using the ``_creflect``
        module. The option is ignored if using the pure python calculator,
        ``_reflect``. If `threads == -1` then all available processors are
        used.

    Example
    -------

    >>> from refnx.reflect import reflectivity
    >>> q = np.linspace(0.01, 0.5, 1000)
    >>> slabs = np.array([[0, 2.07, 0, 0],
    ...                   [100, 3.47, 0, 3],
    ...                   [500, -0.5, 0.00001, 3],
    ...                   [0, 6.36, 0, 3]])
    >>> print(reflectivity(q, slabs))

    """
    # constant dq/q smearing
    if isinstance(dq, numbers.Real) and float(dq) == 0:
        return refcalc.abeles(q, slabs, scale=scale, bkg=bkg, threads=threads)
    elif isinstance(dq, numbers.Real):
        dq = float(dq)
        return (scale *
                _smeared_abeles_constant(q,
                                         slabs,
                                         dq,
                                         threads=threads)) + bkg

    # point by point resolution smearing (each q point has different dq/q)
    if isinstance(dq, np.ndarray) and dq.size == q.size:
        dqvals_flat = dq.flatten()
        qvals_flat = q.flatten()

        # adaptive quadrature
        if quad_order == 'ultimate':
            smeared_rvals = (scale *
                             _smeared_abeles_adaptive(qvals_flat,
                                                      slabs,
                                                      dqvals_flat,
                                                      threads=threads) +
                             bkg)
            return smeared_rvals.reshape(q.shape)
        # fixed order quadrature
        else:
            smeared_rvals = (scale *
                             _smeared_abeles_fixed(qvals_flat,
                                                   slabs,
                                                   dqvals_flat,
                                                   quad_order=quad_order,
                                                   threads=threads) +
                             bkg)
            return np.reshape(smeared_rvals, q.shape)

    # resolution kernel smearing
    elif (isinstance(dq, np.ndarray) and
          dq.ndim == q.ndim + 2 and
          dq.shape[0: q.ndim] == q.shape):

        qvals_for_res = dq[:, 0, :]
        # work out the reflectivity at the kernel evaluation points
        smeared_rvals = refcalc.abeles(qvals_for_res,
                                       slabs,
                                       threads=threads)

        # multiply by probability
        smeared_rvals *= dq[:, 1, :]

        # now do simpson integration
        rvals = scipy.integrate.simps(smeared_rvals, x=dq[:, 0, :])

        return scale * rvals + bkg

    return None


def _memoize_gl(f):
    """
    Cache the gaussian quadrature abscissae, so they don't have to be
    calculated all the time.
    """
    cache = {}

    def inner(n):
        if n in cache:
            return cache[n]
        else:
            result = cache[n] = f(n)
            return result
    return inner


@_memoize_gl
def gauss_legendre(n):
    """
    Calculate gaussian quadrature abscissae and weights
    Parameters
    ----------
    n : int
        Gaussian quadrature order.
    Returns
    -------
    (x, w) : tuple
        The abscissae and weights for Gauss Legendre integration.
    """
    return scipy.special.p_roots(n)


def _smearkernel(x, w, q, dq, threads):
    """
    Adaptive Gaussian quadrature integration

    Parameters
    ----------
    x : float
        Independent variable for integration.
    w : array-like
        The uniform slab model parameters in 'layer' form.
    q : float
        Nominal mean Q of normal distribution
    dq : float
        FWHM of a normal distribution.
    threads : int
        number of threads for parallel calculation
    Returns
    -------
    reflectivity : float
        Model reflectivity multiplied by the probability density function
        evaluated at a given distance, x, away from the mean Q value.
    """
    prefactor = 1 / np.sqrt(2 * np.pi)
    gauss = prefactor * np.exp(-0.5 * x * x)
    localq = q + x * dq / _FWHM
    return refcalc.abeles(localq, w, threads=threads) * gauss


def _smeared_abeles_adaptive(qvals, w, dqvals, threads=-1):
    """
    Resolution smearing that uses adaptive Gaussian quadrature integration
    for the convolution.

    Parameters
    ----------
    qvals : array-like
        The Q values for evaluation
    w : array-like
        The uniform slab model parameters in 'layer' form.
    dqvals : array-like
        dQ values corresponding to each value in `qvals`. Each dqval is the
        FWHM of a Gaussian approximation to the resolution kernel.
    threads : int, optional
        Do you want to calculate in parallel? This option is only applicable if
        you are using the ``_creflect`` module. The option is ignored if using
        the pure python calculator, ``_reflect``.

    Returns
    -------
    reflectivity : np.ndarray
        The smeared reflectivity
    Notes
    -----
    The integration is adaptive meaning it keeps going until it reaches an
    absolute tolerance.
    """
    smeared_rvals = np.zeros(qvals.size)
    warnings.simplefilter('ignore', Warning)
    for idx, val in enumerate(qvals):
        smeared_rvals[idx], err = scipy.integrate.quadrature(
            _smearkernel,
            -_INTLIMIT,
            _INTLIMIT,
            tol=2 * np.finfo(np.float64).eps,
            rtol=2 * np.finfo(np.float64).eps,
            args=(w, qvals[idx], dqvals[idx], threads))

    warnings.resetwarnings()
    return smeared_rvals


def _smeared_abeles_fixed(qvals, w, dqvals, quad_order=17, threads=-1):
    """
    Resolution smearing that uses fixed order Gaussian quadrature integration
    for the convolution.

    Parameters
    ----------
    qvals : array-like
        The Q values for evaluation
    w : array-like
        The uniform slab model parameters in 'layer' form.
    dqvals : array-like
        dQ values corresponding to each value in `qvals`. Each dqval is the
        FWHM of a Gaussian approximation to the resolution kernel.
    quad-order : int, optional
        Specify the order of the Gaussian quadrature integration for the
        convolution.
    threads: int, optional
        Specifies the number of threads for parallel calculation. This
        option is only applicable if you are using the ``_creflect``
        module. The option is ignored if using the pure python calculator,
        ``_reflect``. If `threads == -1` then all available processors are
        used.

    Returns
    -------
    reflectivity : np.ndarray
        The smeared reflectivity
    """

    # The fixed order quadrature does not use scipy.integrate.fixed_quad.
    # That library function does one point at a time, whereas in this function
    # the integration is vectorised

    # get the gauss-legendre weights and abscissae
    abscissa, weights = gauss_legendre(quad_order)

    # get the normal distribution at that point
    prefactor = 1. / np.sqrt(2 * np.pi)

    def gauss(x):
        return np.exp(-0.5 * x * x)

    gaussvals = prefactor * gauss(abscissa * _INTLIMIT)

    # integration between -3.5 and 3.5 sigma
    va = qvals - _INTLIMIT * dqvals / _FWHM
    vb = qvals + _INTLIMIT * dqvals / _FWHM

    va = va[:, np.newaxis]
    vb = vb[:, np.newaxis]

    qvals_for_res = ((np.atleast_2d(abscissa) *
                     (vb - va) + vb + va) / 2.)
    smeared_rvals = refcalc.abeles(qvals_for_res,
                                   w,
                                   threads=threads)

    smeared_rvals = np.reshape(smeared_rvals,
                               (qvals.size, abscissa.size))

    smeared_rvals *= np.atleast_2d(gaussvals * weights)
    return np.sum(smeared_rvals, 1) * _INTLIMIT


def _smeared_abeles_constant(q, w, resolution, threads=-1):
    """
    Fast resolution smearing for constant dQ/Q.

    Parameters
    ----------
    q: np.ndarray
        Q values to evaluate the reflectivity at
    w: np.ndarray
        Parameters for the reflectivity model
    resolution: float
        Percentage dq/q resolution. dq specified as FWHM of a resolution
        kernel.
    threads: int, optional
        Do you want to calculate in parallel? This option is only applicable if
        you are using the ``_creflect`` module. The option is ignored if using
        the pure python calculator, ``_reflect``.
    Returns
    -------
    reflectivity: np.ndarray
        The resolution smeared reflectivity
    """

    if resolution < 0.5:
        return refcalc.abeles(q, w, threads=threads)

    resolution /= 100
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    def gauss(x, s):
        return 1. / s / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2 / s / s)

    lowq = np.min(q)
    highq = np.max(q)
    if lowq <= 0:
        lowq = 1e-6

    start = np.log10(lowq) - 6 * resolution / _FWHM
    finish = np.log10(highq * (1 + 6 * resolution / _FWHM))
    interpnum = np.round(np.abs(1 * (np.abs(start - finish)) /
                         (1.7 * resolution / _FWHM / gaussgpoint)))
    xtemp = np.linspace(start, finish, int(interpnum))
    xlin = np.power(10., xtemp)

    # resolution smear over [-4 sigma, 4 sigma]
    gauss_x = np.linspace(-1.7 * resolution, 1.7 * resolution, gaussnum)
    gauss_y = gauss(gauss_x, resolution / _FWHM)

    rvals = refcalc.abeles(xlin, w, threads=threads)
    smeared_rvals = np.convolve(rvals, gauss_y, mode='same')
    interpolator = InterpolatedUnivariateSpline(xlin, smeared_rvals)

    smeared_output = interpolator(q)
    smeared_output *= gauss_x[1] - gauss_x[0]
    return smeared_output


class MixedReflectModel(object):
    r"""
    Calculates an incoherent average of reflectivities from a sequence of
    structures. Such a situation may occur if a sample is not uniform over its
    illuminated area.

    Parameters
    ----------
    structures : sequence of refnx.reflect.Structure
        The interfacial structures to incoherently average
    scales : None, sequence of float or refnx.analysis.Parameter, optional
        scale factors. The reflectivities calculated from each of the
        structures are multiplied by their respective scale factor during
        overall summation. These values are turned into Parameters during the
        construction of this object.
        You must supply a scale factor for each of the structures. If `scales`
        is `None`, then default scale factors are used:
        `[1 / len(structures)] * len(structures)`. It is a good idea to set the
        lower bound of each scale factor to zero (not done by default).
    bkg : float or refnx.analysis.Parameter, optional
        linear background added to the overall reflectivity. This is turned
        into a Parameter during the construction of this object.
    name : str, optional
        Name of the mixed Model
    dq : float or refnx.analysis.Parameter, optional

        - `dq == 0` then no resolution smearing is employed.
        - `dq` is a float or refnx.analysis.Parameter
           a constant dQ/Q resolution smearing is employed.  For 5% resolution
           smearing supply 5.

        However, if `x_err` is supplied to the `model` method, then that
        overrides any setting given here. This value is turned into
        a Parameter during the construction of this object.
    threads: int, optional
        Specifies the number of threads for parallel calculation. This
        option is only applicable if you are using the ``_creflect``
        module. The option is ignored if using the pure python calculator,
        ``_reflect``. If `threads == -1` then all available processors are
        used.
    quad_order: int, optional
        the order of the Gaussian quadrature polynomial for doing the
        resolution smearing. default = 17. Don't choose less than 13. If
        quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
        quadrature will always work, but takes a _long_ time (2 or 3 orders
        of magnitude longer). Fixed quadrature will always take a lot less
        time. BUT it won't necessarily work across all samples. For
        example, 13 points may be fine for a thin layer, but will be
        atrocious at describing a multilayer with bragg peaks.

    """
    def __init__(self, structures, scales=None, bkg=1e-7, name='', dq=5.,
                 threads=-1, quad_order=17):
        self.name = name
        self._parameters = None
        self.threads = threads
        self.quad_order = quad_order

        # all reflectometry models need a scale factor and background. Set
        # them all to 1 by default.
        pscales = Parameters(name='scale factors')

        if scales is not None and len(structures) == len(scales):
            tscales = scales
        elif scales is not None and len(structures) != len(scales):
            raise ValueError("You need to supply scale factor for each"
                             " structure")
        else:
            tscales = [1 / len(structures)] * len(structures)

        for scale in tscales:
            p_scale = possibly_create_parameter(scale, name='scale')
            pscales.append(p_scale)

        self._scales = pscales
        self._bkg = possibly_create_parameter(bkg, name='bkg')

        # we can optimize the resolution (but this is always overridden by
        # x_err if supplied. There is therefore possibly no dependence on it.
        self._dq = possibly_create_parameter(dq, name='dq - resolution')

        self._structures = structures

    def __repr__(self):
        s = ("MixedReflectModel({_structures!r}, scales={_scales!r},"
             " bkg={_bkg!r}, name={name!r}, dq={_dq!r}, threads={threads!r},"
             " quad_order={quad_order!r})")
        return s.format(**self.__dict__)

    def __call__(self, x, p=None, x_err=None):
        r"""
        Calculate the generative model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity

        """
        return self.model(x, p=p, x_err=x_err)

    @property
    def dq(self):
        r"""
        :class:`refnx.analysis.Parameter`

            - `dq.value == 0`
               no resolution smearing is employed.
            - `dq.value > 0`
               a constant dQ/Q resolution smearing is employed.  For 5%
               resolution smearing supply 5. However, if `x_err` is supplied to
               the `model` method, then that overrides any setting reported
               here.

        """
        return self._dq

    @dq.setter
    def dq(self, value):
        self._dq.value = value

    @property
    def scales(self):
        r"""
        :class:`refnx.analysis.Parameter` - the reflectivity from each of the
        structures are multiplied by these values to account for patchiness.
        """
        return self._scales

    @property
    def bkg(self):
        r"""
        :class:`refnx.analysis.Parameter` - linear background added to all
        model values.

        """
        return self._bkg

    @bkg.setter
    def bkg(self, value):
        self._bkg.value = value

    def model(self, x, p=None, x_err=None):
        r"""
        Calculate the reflectivity of this model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
        p : refnx.analysis.Parameter, optional
            parameters required to calculate the model
        x_err : np.ndarray
            dq resolution smearing values for the dataset being considered.

        Returns
        -------
        reflectivity : np.ndarray

        """
        if p is not None:
            self.parameters.pvals = np.array(p)
        if x_err is None:
            # fallback to what this object was constructed with
            x_err = float(self.dq)

        scales = np.array(self.scales)

        y = np.zeros_like(x)

        for scale, structure in zip(scales, self.structures):
            y += reflectivity(x,
                              structure.slabs()[..., :4],
                              scale=scale,
                              dq=x_err,
                              threads=self.threads,
                              quad_order=self.quad_order)

        return y + self.bkg.value

    def logp(self):
        r"""
        Additional log-probability terms for the reflectivity model. Do not
        include log-probability terms for model parameters, these are
        automatically calculated elsewhere.

        Returns
        -------
        logp : float
            log-probability of structure.

        """
        logp = 0
        for structure in self._structures:
            logp += structure.logp()

        return logp

    @property
    def structures(self):
        r"""
        list of :class:`refnx.reflect.Structure` that describe the patchiness
        of the surface.

        """
        return self._structures

    @property
    def parameters(self):
        r"""
        :class:`refnx.analysis.Parameters` - parameters associated with this
        model.

        """
        p = Parameters(name='instrument parameters')
        p.extend([self.scales, self.bkg, self.dq])

        self._parameters = Parameters(name=self.name)
        self._parameters.append([p])
        self._parameters.extend([structure.parameters for structure
                                 in self._structures])
        return self._parameters
