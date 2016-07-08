from __future__ import division
import warnings
import abc
import math
import numbers
import numpy as np
import scipy
import scipy.linalg
from scipy.interpolate import InterpolatedUnivariateSpline
from refnx.analysis.curvefitter import FitFunction
import refnx.util.ErrorProp as EP
from lmfit import Parameters


try:
    from refnx.analysis import _creflect as refcalc
except ImportError:
    print('WARNING, Using slow reflectivity calculation')
    from refnx.analysis import _reflect as refcalc

# some definitions for resolution smearing
_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5


def coefs_to_layer(coefs):
    """
    Converts 'coefs' format array to a 'layer' format array .
    The 'layer' format is used by :func:`abeles`, the 'coefs' format
    is used by :func:`reflectivity`.
    The 'layer' format has N + 2 rows and 4 columns. Each row describes a
    separate layer in the model. The 4 columns describe the thickness, SLD,
    iSLD and roughness of each layer.
    The 'coefs' format is a vector description of the same information. A
    vector form is required for fitting purposes.

    Parameters
    ----------
    coefs : np.ndarray

        * coefs[0] = number of layers, N
        * coefs[1] = scale factor
        * coefs[2] = SLD of fronting (/1e-6 Angstrom**-2)
        * coefs[3] = iSLD of fronting (/1e-6 Angstrom**-2)
        * coefs[4] = SLD of backing
        * coefs[5] = iSLD of backing
        * coefs[6] = background
        * coefs[7] = roughness between backing and layer N

        * coefs[4 * (N - 1) + 8] = thickness of layer N in Angstrom (layer 1 is
        * closest to fronting)
        * coefs[4 * (N - 1) + 9] = SLD of layer N (/ 1e-6 Angstrom**-2)
        * coefs[4 * (N - 1) + 10] = iSLD of layer N (/ 1e-6 Angstrom**-2)
        * coefs[4 * (N - 1) + 11] = roughness between layer N and N-1.

    Returns
    -------
    layers: np.ndarray

        Has shape (2 + N, 4), where N is the number of layers

        * layers[0, 1] - SLD of fronting (/ 1e-6 Angstrom**-2)
        * layers[0, 2] - iSLD of fronting (/ 1e-6 Angstrom**-2)
        * layers[N, 0] - thickness of layer N
        * layers[N, 1] - SLD of layer N (/ 1e-6 Angstrom**-2)
        * layers[N, 2] - iSLD of layer N (/ 1e-6 Angstrom**-2)
        * layers[N, 3] - roughness between layer N-1/N
        * layers[-1, 1] - SLD of backing (/ 1e-6 Angstrom**-2)
        * layers[-1, 2] - iSLD of backing (/ 1e-6 Angstrom**-2)
        * layers[-1, 3] - roughness between backing and last layer

    """
    nlayers = int(coefs[0])
    layers = np.zeros((nlayers + 2, 4), np.float64)
    layers[0, 1: 3] = coefs[2: 4]
    layers[-1, 1: 3] = coefs[4: 6]
    layers[-1, 3] = coefs[7]
    if nlayers:
        layers[1:-1] = np.array(coefs[8:]).reshape(nlayers, 4)

    return layers


def layer_to_coefs(layers, scale=1, bkg=0):
    r"""
    Converts 'layer' format array to a 'coefs' format array .
    The 'layer' format is used by the :func:`abeles` function,
    the 'coefs' format is used by the :func:`reflectivity` function.
    The 'layer' format has N + 2 rows and 4 columns. Each row describes a
    separate layer in the model. The 4 columns describe the thickness, SLD,
    iSLD and roughness of each layer.
    The 'coefs' format is a vector description of the same information. A
    vector form is required for fitting purposes.

    Parameters
    ----------
    layers : np.ndarray
        Has shape (2 + N, 4), where N is the number of layers.

        * layers[0, 1] = SLD of fronting (/ 1e-6 Angstrom**-2)
        * layers[0, 2] = iSLD of fronting (/ 1e-6 Angstrom**-2)
        * layers[N, 0] = thickness of layer N
        * layers[N, 1] = SLD of layer N (/ 1e-6 Angstrom**-2)
        * layers[N, 2] = iSLD of layer N (/ 1e-6 Angstrom**-2)
        * layers[N, 3] = roughness between layer N-1/N
        * layers[-1, 1] = SLD of backing (/ 1e-6 Angstrom**-2)
        * layers[-1, 2] = iSLD of backing (/ 1e-6 Angstrom**-2)
        * layers[-1, 3] = roughness between backing and last layer

    Returns
    -------
    coefs : np.ndarray
        Has shape (4 * N + 8, ), where N is the number of layers

        * coefs[0] = number of layers, N
        * coefs[1] = scale factor
        * coefs[2] = SLD of fronting (/1e-6 Angstrom**-2)
        * coefs[3] = iSLD of fronting (/1e-6 Angstrom**-2)
        * coefs[4] = SLD of backing
        * coefs[5] = iSLD of backing
        * coefs[6] = background
        * coefs[7] = roughness between backing and layer N
        * coefs[4 * (N - 1) + 8] = thickness of layer N in Angstrom (layer 1 is
          closest to fronting)
        * coefs[4 * (N - 1) + 9] = SLD of layer N (/ 1e-6 Angstrom**-2)
        * coefs[4 * (N - 1) + 10] = iSLD of layer N (/ 1e-6 Angstrom**-2)
        * coefs[4 * (N - 1) + 11] = roughness between layer N and N-1.

    """

    nlayers = np.size(layers, 0) - 2
    coefs = np.zeros(4 * nlayers + 8, np.float64)
    coefs[0] = nlayers
    coefs[1] = scale
    coefs[2:4] = layers[0, 1: 3]
    coefs[4: 6] = layers[-1, 1: 3]
    coefs[6] = bkg
    coefs[7] = layers[-1, 3]
    if nlayers:
        coefs[8:] = layers.ravel()[4: -4]

    return coefs


def abeles(q, layers, scale=1, bkg=0., parallel=True):
    r"""
    Abeles matrix formalism for calculating reflectivity from a stratified
    medium.

    Parameters
    ----------
    q : array_like
        the q values required for the calculation.
        :math:`Q = \frac{4\pi}{\lambda}\sin(\Omega)`.
        Units = Angstrom**-1
    layers : np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers

        * layers[0, 1] - SLD of fronting (/ 1e-6 Angstrom**-2)
        * layers[0, 2] - iSLD of fronting (/ 1e-6 Angstrom**-2)
        * layers[N, 0] - thickness of layer N
        * layers[N, 1] - SLD of layer N (/ 1e-6 Angstrom**-2)
        * layers[N, 2] - iSLD of layer N (/ 1e-6 Angstrom**-2)
        * layers[N, 3] - roughness between layer N-1/N
        * layers[-1, 1] - SLD of backing (/ 1e-6 Angstrom**-2)
        * layers[-1, 2] - iSLD of backing (/ 1e-6 Angstrom**-2)
        * layers[-1, 3] - roughness between backing and last layer

    scale : float
        Multiply all reflectivities by this value.
    bkg : float
        Linear background to be added to all reflectivities
    parallel : bool
        Do you want to calculate in parallel? This option is only applicable if
        you are using the ``_creflect`` module. The option is ignored if using
        the pure python calculator, ``_reflect``.

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """
    return refcalc.abeles(q, layers, scale=scale, bkg=bkg, parallel=parallel)


def reflectivity(q, coefs, *args, **kwds):
    r"""
    Abeles matrix formalism for calculating reflectivity from a stratified
    medium.

    Parameters
    ----------
    q : np.ndarray
        The qvalues required for the calculation. :math:`Q=\frac{4Pi}{\lambda}\sin(\Omega)`.
        Units = Angstrom**-1
    coefs : np.ndarray

        * coefs[0] - number of layers, N
        * coefs[1] - scale factor
        * coefs[2] - SLD of fronting (/1e-6 Angstrom**-2)
        * coefs[3] - iSLD of fronting (/1e-6 Angstrom**-2)
        * coefs[4] - SLD of backing
        * coefs[5] - iSLD of backing
        * coefs[6] - background
        * coefs[7] - roughness between backing and layer N
        * coefs[4 * (N - 1) + 8] = thickness of layer N in Angstrom (layer 1 is
          closest to fronting)
        * coefs[4 * (N - 1) + 9] - SLD of layer N (/ 1e-6 Angstrom**-2)
        * coefs[4 * (N - 1) + 10] - iSLD of layer N (/ 1e-6 Angstrom**-2)
        * coefs[4 * (N - 1) + 11] - roughness between layer N and N-1.

    kwds : dict, optional
        The following keys are used:

        'dqvals': float or np.ndarray, optional
            If dqvals is a float, then a constant dQ/Q resolution smearing is
            employed.  For 5% resolution smearing supply 5.
            If `dqvals` is the same shape as q, then the array contains the
            FWHM of a Gaussian approximated resolution kernel. Point by point
            resolution smearing is employed.  Use this option if dQ/Q varies
            across your dataset.
            If `dqvals.ndim == q.ndim + 2` and
            `q.shape == dqvals[..., -3].shape` then an individual resolution
            kernel is applied to each measurement point.  This resolution kernel
            is a probability distribution function (PDF). `dqvals` will have the
            shape (qvals.shape, M, 2).  There are `M` points in the kernel.
            `dqvals[..., 0]` holds the q values for the kernel, `dqvals[..., 1]`
            gives the corresponding probability.
        'quad_order': int, optional
            the order of the Gaussian quadrature polynomial for doing the
            resolution smearing. default = 17. Don't choose less than 13. If
            quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
            quadrature will always work, but takes a _long_ time (2 or 3 orders
            of magnitude longer). Fixed quadrature will always take a lot less
            time. BUT it won't necessarily work across all samples. For example,
            13 points may be fine for a thin layer, but will be atrocious at
            describing a multilayer with bragg peaks.
        'parallel': bool, optional
            Do you want to calculate in parallel? This option is only
            applicable if you are using the ``_creflect`` module. The option is
            ignored if using the pure python calculator, ``_reflect``. The
            default is `True`.

    """
    parallel=True
    if 'parallel' in kwds:
        parallel = kwds['parallel']

    qvals = q
    quad_order = 17
    scale = coefs[1]
    bkg = coefs[6]

    if not is_proper_abeles_input(coefs):
        raise ValueError('The size of the parameter array passed to reflectivity'
                         ' should be 4 * coefs[0] + 8')

    # make into form suitable for reflection calculation
    w = coefs_to_layer(coefs)

    if 'quad_order' in kwds:
        quad_order = kwds['quad_order']

    if 'dqvals' in kwds and kwds['dqvals'] is not None:
        dqvals = kwds['dqvals']

        # constant dq/q smearing
        if isinstance(dqvals, numbers.Real):
            dqvals = float(dqvals)
            return (scale * _smeared_abeles_constant(qvals,
                                                     w,
                                                     dqvals,
                                                     parallel=parallel)) + bkg

        # point by point resolution smearing
        if dqvals.size == qvals.size:
            dqvals_flat = dqvals.flatten()
            qvals_flat = q.flatten()

            # adaptive quadrature
            if quad_order == 'ultimate':
                smeared_rvals = (scale *
                    _smeared_abeles_adaptive(qvals_flat,
                                             w,
                                             dqvals_flat,
                                             parallel=parallel) + bkg)
                return smeared_rvals.reshape(q.shape)
            # fixed order quadrature
            else:
                smeared_rvals = (scale * _smeared_abeles_fixed(
                                                      qvals_flat,
                                                      w,
                                                      dqvals_flat,
                                                      quad_order=quad_order,
                                                      parallel=parallel)
                                 + bkg)
                return np.reshape(smeared_rvals, q.shape)

        # resolution kernel smearing
        elif (dqvals.ndim == qvals.ndim + 2
              and dqvals.shape[0: qvals.ndim] == qvals.shape):
            # TODO may not work yet.
            qvals_for_res = dqvals[..., 0]
            # work out the reflectivity at the kernel evaluation points
            smeared_rvals = refcalc.abeles(qvals_for_res,
                                           w,
                                           scale=coefs[1],
                                           bkg=coefs[6],
                                           parallel=parallel)

            # multiply by probability
            smeared_rvals *= dqvals[..., 1]

            # now do simpson integration
            return scipy.integrate.simps(smeared_rvals, x=dqvals[..., 0])

    # no smearing
    return refcalc.abeles(q,
                          w,
                          scale=coefs[1],
                          bkg=coefs[6],
                          parallel=parallel)


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
    k = np.arange(1.0, n)
    a_band = np.zeros((2, n))
    a_band[1, 0: n - 1] = k / np.sqrt(4 * k * k - 1)
    x, v = scipy.linalg.eig_banded(a_band, lower=True)
    w = 2 * np.real(np.power(v[0, :], 2))
    return x, w


def _smearkernel(x, w, q, dq, parallel):
    """
    Kernel for adaptive Gaussian quadrature integration

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

    Returns
    -------
    reflectivity : float
        Model reflectivity multiplied by the probability density function
        evaluated at a given distance, x, away from the mean Q value.
    """
    prefactor = 1 / np.sqrt(2 * np.pi)
    gauss = prefactor * np.exp(-0.5 * x * x)
    localq = q + x * dq / _FWHM
    return refcalc.abeles(localq, w, parallel=parallel) * gauss


def _smeared_abeles_adaptive(qvals, w, dqvals, parallel=True):
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
    parallel: bool, optional
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
            args=(w, qvals[idx], dqvals[idx], parallel))

    warnings.resetwarnings()
    return smeared_rvals


def _smeared_abeles_fixed(qvals, w, dqvals, quad_order=17, parallel=True):
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
    parallel: bool, optional
        Do you want to calculate in parallel? This option is only applicable if
        you are using the ``_creflect`` module. The option is ignored if using
        the pure python calculator, ``_reflect``.


    Returns
    -------
    reflectivity : np.ndarray
        The smeared reflectivity
    """
    # get the gauss-legendre weights and abscissae
    abscissa, weights = gauss_legendre(quad_order)

    # get the normal distribution at that point
    prefactor = 1. / np.sqrt(2 * np.pi)
    gauss = lambda x: np.exp(-0.5 * x * x)
    gaussvals = prefactor * gauss(abscissa * _INTLIMIT)

    # integration between -3.5 and 3.5 sigma
    va = qvals - _INTLIMIT * dqvals / _FWHM
    vb = qvals + _INTLIMIT * dqvals / _FWHM

    va = va[:, np.newaxis]
    vb = vb[:, np.newaxis]

    qvals_for_res = ((np.atleast_2d(abscissa) *
                     (vb - va)
                     + vb + va) / 2.)
    smeared_rvals = refcalc.abeles(qvals_for_res.flatten(),
                                   w,
                                   parallel=parallel)

    smeared_rvals = np.reshape(smeared_rvals,
                               (qvals.size, abscissa.size))

    smeared_rvals *= np.atleast_2d(gaussvals * weights)
    return np.sum(smeared_rvals, 1) * _INTLIMIT


def _smeared_abeles_constant(q, w, resolution, parallel=True):
    """
    A kernel for fast and constant dQ/Q smearing

    Parameters
    ----------
    q: np.ndarray
        Q values to evaluate the reflectivity at
    w: np.ndarray
        Parameters for the reflectivity model
    resolution: float
        Percentage dq/q resolution. dq specified as FWHM of a resolution
        kernel.
    parallel: bool, optional
        Do you want to calculate in parallel? This option is only applicable if
        you are using the ``_creflect`` module. The option is ignored if using
        the pure python calculator, ``_reflect``.


    Returns
    -------
    reflectivity: np.ndarray
        The resolution smeared reflectivity
    """

    if resolution < 0.5:
        return refcalc.abeles(q, w, parallel=parallel)

    resolution /= 100
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    gauss = lambda x, s: (1. / s / np.sqrt(2 * np.pi)
                          * np.exp(-0.5 * x**2 / s / s))

    lowq = np.min(q)
    highq = np.max(q)
    if lowq <= 0:
        lowq = 1e-6

    start = np.log10(lowq) - 6 * resolution / _FWHM
    finish = np.log10(highq * (1 + 6 * resolution / _FWHM))
    interpnum = np.round(np.abs(1 * (np.abs(start - finish))
                                / (1.7 * resolution / _FWHM / gaussgpoint)))
    xtemp = np.linspace(start, finish, int(interpnum))
    xlin = np.power(10., xtemp)

    # resolution smear over [-4 sigma, 4 sigma]
    gauss_x = np.linspace(-1.7 * resolution, 1.7 * resolution, gaussnum)
    gauss_y = gauss(gauss_x, resolution / _FWHM)

    rvals = refcalc.abeles(xlin, w, parallel=parallel)
    smeared_rvals = np.convolve(rvals, gauss_y, mode='same')
    interpolator = InterpolatedUnivariateSpline(xlin, smeared_rvals)

    smeared_output = interpolator(q)
    # smeared_output *= np.sum(gauss_y)
    smeared_output *= gauss_x[1] - gauss_x[0]
    return smeared_output


def is_proper_abeles_input(coefs):
    """
    Test to see if the coefs array is suitable input for the `reflectivity`
    function

    Parameters
    ----------
    coefs : np.ndarray
        Coefficients used for calculating reflectivity, as passed to
        :func:`reflectivity`
    Returns
    -------
    is_proper_abeles_input : bool
        Truth of whether the coeffcients have the right number of parameters
        for the number of layers in the model.
    """
    if np.size(coefs, 0) != 4 * int(coefs[0]) + 8:
        return False
    return True


def sld_profile(z, coefs):
    """
    Calculates an SLD profile, as a function of distance through the
    interface.

    Parameters
    ----------
    z : float
        Interfacial distance (Angstrom) measured from interface between the
        fronting medium and the first layer.
    coefs : np.ndarray
        The reflectivity model parameters in 'layer' form. (See
        `reflectivity`)

    Returns
    -------
    sld : float
        Scattering length density / 1e-6 $\AA^-2$

    Notes
    -----
    This can be called in vectorised fashion.
    """
    nlayers = int(coefs[0])
    sld = np.zeros_like(z)
    sld += coefs[2]
    thick = 0

    for idx, zed in enumerate(z):
        dist = 0
        for ii in range(nlayers + 1):
            if ii == 0:
                if nlayers:
                    deltarho = -coefs[2] + coefs[9]
                    thick = 0
                    sigma = np.fabs(coefs[11])
                else:
                    sigma = np.fabs(coefs[7])
                    deltarho = -coefs[2] + coefs[4]
            elif ii == nlayers:
                sld1 = coefs[4 * ii + 5]
                deltarho = -sld1 + coefs[4]
                thick = np.fabs(coefs[4 * ii + 4])
                sigma = np.fabs(coefs[7])
            else:
                sld1 = coefs[4 * ii + 5]
                sld2 = coefs[4 * (ii + 1) + 5]
                deltarho = -sld1 + sld2
                thick = np.fabs(coefs[4 * ii + 4])
                sigma = np.fabs(coefs[4 * (ii + 1) + 7])

            dist += thick

            # if sigma=0 then the computer goes haywire (division by zero), so
            # say it's vanishingly small
            if sigma == 0:
                sigma += 1e-3

            # summ += deltarho * (norm.cdf((zed - dist)/sigma))
            sld[idx] += (deltarho *
                (0.5 + 0.5 * math.erf((zed - dist) / (sigma * np.sqrt(2.)))))

    return sld


class ReflectivityFitFunction(FitFunction):
    """
    A sub class of `refnx.analysis.curvefitter.FitFunction` suited for
    calculation of reflectometry profiles from a simple slab model

    Parameters
    ----------
    transform : callable, optional
        If specified then this function is used to transform the data
        returned by the model method. With the signature:
        ``transformed_y_vals = transform(x_vals, y_vals)``.
    dq : float, optional
        Default dq/q resolution (as a percentage).
    quad_order : int or str, optional
        The order of the Gaussian quadrature polynomial for doing the
        resolution smearing. default = 17. Don't choose less than 13. If
        quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
        quadrature will always work, but takes a _long_ time (2 or 3 orders
        of magnitude longer). Fixed quadrature will always take a lot less
        time. BUT it won't necessarily work across all samples. For
        example 13 points may be fine for a thin layer, but will be
        atrocious at describing a multilayer with Bragg peaks.
    parallel : bool, optional
        Do you want to calculate in parallel? This option is only
        applicable if you are using the ``_creflect`` module. The option is
        ignored if using the pure python calculator, ``_reflect``.
    """

    def __init__(self, transform=None, dq=5., quad_order=17, parallel=True):
        """
        Initialises the ReflectivityFitFunction.
        """
        super(ReflectivityFitFunction, self).__init__()

        self.transform = transform
        self.dq = float(dq)
        self.quad_order = quad_order
        self.parallel = parallel

    def model(self, x, parameters, *args, **kwds):
        """
        Calculate the theoretical model, given a set of parameters.

        Parameters
        ----------
        x : array-like
            Q values to evaluate the reflectivity at
        parameters : lmfit.Parameters instance or sequence
            Contains the parameters that are required for reflectivity
            calculation. See ``reflectivity`` for the required parameters for
            calculation
        kwds['dqvals'] : float or np.ndarray, optional
            If dqvals is a float, then a constant dQ/Q resolution smearing is
            employed.  For 5% resolution smearing supply 5.
            If `dqvals` is the same shape as q, then the array contains the
            FWHM of a Gaussian approximated resolution kernel. Point by point
            resolution smearing is employed.  Use this option if dQ/Q varies
            across your dataset.
            If `dqvals.ndim == q.ndim + 2` and
            `q.shape == dqvals[..., -3].shape` then an individual resolution
            kernel is applied to each measurement point.  This resolution kernel
            is a probability distribution function (PDF). `dqvals` will have the
            shape (qvals.shape, M, 2).  There are `M` points in the kernel.
            `dqvals[..., 0]` holds the q values for the kernel, `dqvals[..., 1]`
            gives the corresponding probability.
        kwds['quad_order'] : int, optional
            the order of the Gaussian quadrature polynomial for doing the
            resolution smearing. default = 17. Don't choose less than 13. If
            quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
            quadrature will always work, but takes a _long_ time (2 or 3 orders
            of magnitude longer). Fixed quadrature will always take a lot less
            time. BUT it won't necessarily work across all samples. For example,
            13 points may be fine for a thin layer, but will be atrocious at
            describing a multilayer with bragg peaks.
        kwds['parallel'] : bool, optional
            Do you want to calculate in parallel? This option is only
            applicable if you are using the ``_creflect`` module. The option is
            ignored if using the pure python calculator, ``_reflect``. The
            default is `True`.

        Returns
        -------
        y : np.ndarray
            The predictive model, i.e.
            ``reflectivity(x, parameters, *args, **kwds)``
        """
        if isinstance(parameters, Parameters):
            params = np.array([parameters[param].value for param in parameters],
                              float)
        else:
            params = parameters

        if not 'quad_order' in kwds:
            kwds['quad_order'] = self.quad_order
        if not 'dqvals' in kwds and self.dq > 0.3:
            kwds['dqvals'] = float(self.dq)
        if not 'parallel' in kwds:
            kwds['parallel'] = self.parallel
        yvals = reflectivity(x, params, *args, **kwds)

        if self.transform or 'transform' in kwds:
            t = self.transform or kwds['transform']
            yvals, temp = t(x, yvals)

        return yvals

    def set_dq(self, dq, quad_order=17):
        """
        Sets the resolution information.

        Parameters
        ----------
        dq : None, float or np.ndarray
            If `None` then there is no resolution smearing.
            If a float, e.g. 5, then dq/q smearing of 5% is applied. If dq==0
            then resolution smearing is removed.
            If an np.ndarray the same length as y, it contains the FWHM of
            the Gaussian approximated resolution kernel.
        quad_order : int or 'ultimate'
            The order of the Gaussian quadrature polynomial for doing the
            resolution smearing. default = 17. Don't choose less than 13. If
            quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
            quadrature will always work, but takes a _long_ time (2 or 3 orders
            of magnitude longer). Fixed quadrature will always take a lot less
            time. BUT it won't necessarily work across all samples. For
            example 13 points may be fine for a thin layer, but will be
            atrocious at describing a multilayer with Bragg peaks.
        """
        if type(dq) is float or type(dq) is int:
            self.dq = float(dq)

        if self.dq < 0.3:
            self.dq = 0.

        if type(quad_order) is int and quad_order > 12:
            self.quad_order = quad_order
        elif quad_order == 'ultimate':
            self.quad_order = 'ultimate'

    def sld_profile(self, parameters, z=None):
        """
        Calculate the SLD profile corresponding to the model parameters.

        Parameters
        ----------
        parameters : lmfit.parameters.Parameters instance or sequence

        z : array-like, optional
            Interfacial distances to evaluate the SLD profile at.
            z = 0 corresponds to the interfaces between the fronting medium
            and the first layer

        Returns
        -------
        (z, rho_z) : tuple of np.ndarrays
            The distance from the top interface and the SLD at that point.
        """
        if isinstance(parameters, Parameters):
            params = np.array([parameters[param].value for param in parameters],
                              float)
        else:
            params = parameters

        if z is not None:
            return z, sld_profile(z, params)

        if not int(params[0]):
            zstart = -5 - 4 * np.fabs(params[7])
        else:
            zstart = -5 - 4 * np.fabs(params[11])

        temp = 0
        if not int(params[0]):
            zend = 5 + 4 * np.fabs(params[7])
        else:
            for i in range(int(params[0])):
                temp += np.fabs(params[4 * i + 8])
            zend = 5 + temp + 4 * np.fabs(params[7])

        z = np.linspace(zstart, zend, num=500)

        return z, sld_profile(z, params)

    @staticmethod
    def parameter_names(nparams=8):
        """
        Parameter names for a default reflectivity calculation
        """
        names = ['nlayers', 'scale', 'SLDfront', 'iSLDfront', 'SLDback',
                 'iSLDback', 'bkg', 'sigma_back']
        nlayers = (nparams - 8) / 4
        for i in range(int(nlayers)):
            names.append('thick%d' % (i + 1))
            names.append('SLD%d' % (i + 1))
            names.append('iSLD%d' % (i + 1))
            names.append('sigma%d' % (i + 1))
        return names

    def callback(self, parameters, iteration, resid, *fcn_args, **fcn_kws):
        return False


class AnalyticalReflectivityFunction(ReflectivityFitFunction):
    """
    A class for using analytical profiles in Reflectometry problems
    Usage involves inheriting this class and over-riding ``to_slab`` and
    ``parameter_names``.
    """
    def __init__(self, transform=None, dq=5., quad_order=17, parallel=True):
        """
        Parameters
        ----------
        transform : callable, optional
            If specified then this function is used to transform the data
            returned by the model method. With the signature:
            ``transformed_y_vals = transform(x_vals, y_vals)``.
        dq : float, optional
            Default dq/q resolution (as a percentage).
        quad_order : int or str, optional
            The order of the Gaussian quadrature polynomial for doing the
            resolution smearing. default = 17. Don't choose less than 13. If
            quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
            quadrature will always work, but takes a _long_ time (2 or 3 orders
            of magnitude longer). Fixed quadrature will always take a lot less
            time. BUT it won't necessarily work across all samples. For
            example 13 points may be fine for a thin layer, but will be
            atrocious at describing a multilayer with Bragg peaks.
        parallel : bool, optional
            Do you want to calculate in parallel? This option is only
            applicable if you are using the ``_creflect`` module. The option is
            ignored if using the pure python calculator, ``_reflect``.
        """
        s_klass = super(AnalyticalReflectivityFunction, self)
        s_klass.__init__(transform=transform, dq=dq, quad_order=quad_order,
                         parallel=parallel)

    def model(self, x, parameters, *args, **kwds):
        """
        Calculates the reflectivity model. You should not need to over-ride
        this method.
        """
        slab_pars = self.to_slab(parameters)
        s_klass = super(AnalyticalReflectivityFunction, self)
        return s_klass.model(x, slab_pars, *args, **kwds)

    def sld_profile(self, parameters, z=None):
        """
        Calculates the SLD profile. You should not need to over-ride
        this method.
        """
        slab_pars = self.to_slab(parameters)
        s_klass = super(AnalyticalReflectivityFunction, self)
        return s_klass.sld_profile(slab_pars, z=z)

    @abc.abstractmethod
    def to_slab(self, params):
        """
        Maps Parameters from your analytical model to those suitable
        for a simple slab reflectivity calculation. See ``reflectivity`` for
        the correct output format.

        Parameters
        ----------
        params : lmfit.Parameters or sequence
            Parameters specifying your analytical model

        Returns
        -------
        slab_params : lmfit.Parameters or sequence
            Parameters usable for simple slab reflectivity calculation. See
            :func:`reflectivity` for the correct format for slab_params. Should
            have: `len(slab_params) == 4 * slab_params[0] + 8`.
        """
        pass

    @abc.abstractmethod
    def parameter_names(self, nparams=None):
        """
        Specifies the names of the parameters for this analytical model

        Parameters
        ----------
        nparams : int
            Number of parameters

        Returns
        -------
        names : sequence
            List containing the names of each of the parameters in this model
        """
        pass


class Transform(object):
    r"""
    Mathematical transforms of numeric data

    Parameters
    ----------
    form : None or str
        One of:

            - 'lin'
                No transform is made
            - 'logY'
                log10 transform
            - 'YX4'
                YX**4 transform
            - 'YX2'
                YX**2 transform
            - None
                No transform is made
    """
    def __init__(self, form):
        types = [None, 'lin', 'logY', 'YX4', 'YX2']
        self.form = None

        if form in types:
            self.form = form

    def transform(self, xdata, ydata, edata=None):
        r"""
        Transform the data passed in

        Parameters
        ----------
        xdata : array-like

        ydata : array-like

        edata : array-like

        Returns
        -------
        yt, et : tuple
            The transformed data
        """

        if edata is None:
            etemp = np.ones_like(ydata)
        else:
            etemp = edata

        if self.form == None:
            yt = np.copy(ydata)
            et = np.copy(etemp)
        elif self.form == 'lin':
            yt = np.copy(ydata)
            et = np.copy(etemp)
        elif self.form == 'logY':
            yt, et = EP.EPlog10(ydata, etemp)
        elif self.form == 'YX4':
            yt = ydata * np.power(xdata, 4)
            et = etemp * np.power(xdata, 4)
        elif self.form == 'YX2':
            yt = ydata * np.power(xdata, 2)
            et = etemp * np.power(xdata, 2)
        if edata is None:
            return yt, None
        else:
            return yt, et


if __name__ == '__main__':
    import timeit
    a = np.zeros((12))
    a[0] = 1.
    a[1] = 1.
    a[4] = 2.07
    a[7] = 3
    a[8] = 100
    a[9] = 3.47
    a[11] = 2

    b = np.arange(10000.)
    b /= 20000.
    b += 0.0005

    def _loop():
        reflectivity(b, a)

    t = timeit.Timer(stmt=_loop)
    print(t.timeit(number=1000))
