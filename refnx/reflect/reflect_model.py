"""
*Calculates the specular (Neutron or X-ray) reflectivity from a stratified
series of layers.

The refnx code is distributed under the following license:

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
from contextlib import contextmanager
import time
from functools import lru_cache
import numbers
import warnings

import numpy as np
import scipy
from scipy.interpolate import splrep, splev


from refnx.analysis import (
    Parameters,
    Parameter,
    possibly_create_parameter,
    Transform,
)


# some definitions for resolution smearing
_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5


"""
Implementation notes
--------------------
1. For _smeared_abeles_fixed I investigated calculating a master curve,
   adjacent data points have overlapping resolution kernels. So instead of
   using e.g. an oversampling factor of 17, one could get away with using
   a factor of 6. This is because the calculated points can be used to smear
   more than one datapoint. One can't use Gaussian quadrature, Simpsons rule is
   needed. Technically the approach works, but turns out to be no faster than
   the Gaussian quadrature with the x17 oversampling (even if everything is
   vectorised). There are a couple of reasons:
   a) calculating the Gaussian weights has to be re-done for all the resolution
   smearing points for every datapoint. For Gaussian quadrature that
   calculation only needs to be done once, because the oversampling points are
   at constant locations around the mean. I experimented with using a cached
   spline to evaluate the Gaussian weights (rather than explicit calculation),
   but this is no faster.
   b) in the implementation I tried the Simpsons rule had to integrate
   e.g. 700 odd points instead of the fixed 17 for the Gaussian quadrature.
"""


def available_backends():
    """
    tuple containing the available reflectivity calculator backends
    """
    backends = ["python"]
    try:
        import refnx.reflect._creflect as _creflect

        backends.append("c")
    except ImportError:
        pass

    try:
        import refnx.reflect._cyreflect as _cyreflect

        backends.append("cython")
    except ImportError:
        pass

    try:
        import pyopencl as cl

        cl.get_platforms()
        from refnx.reflect._reflect import abeles_pyopencl

        backends.append("pyopencl")
    except Exception:
        # importing pyopencl would be a ModuleNotFoundError
        # failure to get an opencl platform would be cl._cl.LogicError
        pass

    # try:
    #     import jax as jax
    #     from jax.config import config
    #
    #     config.update("jax_enable_x64", True)
    #     from refnx.reflect._jax_reflect import abeles_jax
    #
    #     backends.append("jax")
    # except Exception:
    #     # importing jax would be a ModuleNotFoundError
    #     pass

    return tuple(backends)


def get_reflect_backend(backend="c"):
    r"""
    Obtain an 'abeles' function used for calculating reflectivity.

    It does not change the function used by ReflectModel to calculate
    reflectivity. In order to change this you should change the
    refnx.reflect.reflect_model.abeles module variable to point to a different
    function, or use the refnx.reflect.reflect_model.use_reflect_backend
    context manager.

    Parameters
    ----------
    backend: {'python', 'cython', 'c', 'pyopencl'}, str
        The module that calculates the reflectivity. Speed should go in the
        order: c > pyopencl / cython > python. If a particular method is
        not available the function falls back:
        cython/pyopencl --> c --> python.

    Returns
    -------
    abeles: callable
        The callable that calculates the reflectivity

    Notes
    -----
    'c' is preferred for most circumstances.
    'pyopencl' uses a GPU to calculate reflectivity and requires that pyopencl
    be installed. It may not as accurate as the other options. 'pyopencl' is
    only included for completeness. The 'pyopencl' backend is also harder to
    use with multiprocessing-based parallelism.
    """
    backend = backend.lower()

    if backend == "pyopencl":
        try:
            import pyopencl as cl
        except (ImportError, ModuleNotFoundError):
            warnings.warn(
                "Can't use the pyopencl abeles backend, you need"
                " to install pyopencl"
            )
            return get_reflect_backend("c")
        try:
            # see if there are any openCL platforms
            cl.get_platforms()
            from refnx.reflect._reflect import abeles_pyopencl

            return abeles_pyopencl
        except cl._cl.LogicError:
            # a pyopencl._cl.LogicError is raised if there isn't a platform
            warnings.warn("There are no openCL platforms available")
            return get_reflect_backend("c")
    elif backend == "cython":
        try:
            from refnx.reflect import _cyreflect as _cy

            return _cy.abeles
        except ImportError:
            warnings.warn("Can't use the cython abeles backend")
            return get_reflect_backend("c")
    elif backend == "c":
        try:
            from refnx.reflect import _creflect as _c

            return _c.abeles
        except ImportError:
            warnings.warn("Can't use the C abeles backend")
            return get_reflect_backend("python")
    # elif backend == "jax":
    #     try:
    #         from refnx.reflect import _jax_reflect
    #
    #         return _jax_reflect.abeles_jax
    #     except ImportError:
    #         warnings.warn("Can't use the jax abeles backend")
    #         return get_reflect_backend("c")

    elif backend == "python":
        warnings.warn("Using the SLOW reflectivity calculation.")

    # if nothing works return the Python backend
    from refnx.reflect import _reflect as _py

    return _py.abeles


# this function is used to calculate reflectivity
abeles = get_reflect_backend("c")


@contextmanager
def use_reflect_backend(backend="c"):
    """Context manager for temporarily setting the backend used for
    calculating reflectivity

    Parameters
    ----------
    backend: {'python', 'cython', 'c', 'pyopencl'}, str
        The function that calculates the reflectivity. Speed should go in the
        order: c > pyopencl / cython > python. If a particular method is not
        available the function falls back: cython/pyopencl --> c --> python.

    Yields
    ------
    abeles: callable
        A callable that calculates the reflectivity

    Notes
    -----
    'c' is preferred for most circumstances.
    'pyopencl' uses a GPU to calculate reflectivity and requires that pyopencl
    be installed. It may not as accurate as the other options. 'pyopencl' is
    only included for completeness. The 'pyopencl' backend is also harder to
    use with multiprocessing-based parallelism.
    """
    global abeles
    f = abeles
    abeles = get_reflect_backend(backend)
    yield abeles
    abeles = f


class ReflectModel:
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

        This value is turned into a Parameter during the construction of this
        object.
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
    dq_type: {'pointwise', 'constant'}, optional
        Chooses whether pointwise or constant dQ/Q resolution smearing (see
        `dq` keyword) is used. To use pointwise smearing the `x_err` keyword
        provided to `Objective.model` method must be an array, otherwise the
        smearing falls back to 'constant'.
    q_offset: float or refnx.analysis.Parameter, optional
        Compensates for uncertainties in the angle at which the measurement is
        performed. A positive/negative `q_offset` corresponds to a situation
        where the measured q values (incident angle) may have been under/over
        estimated, and has the effect of shifting the calculated model to
        lower/higher effective q values.
    """

    def __init__(
        self,
        structure,
        scale=1,
        bkg=1e-7,
        name="",
        dq=5.0,
        threads=-1,
        quad_order=17,
        dq_type="pointwise",
        q_offset=0,
    ):
        self.name = name
        self._parameters = None
        self.threads = threads
        self.quad_order = quad_order

        # to make it more like a refnx.analysis.Model
        self.fitfunc = None

        # all reflectometry models need a scale factor and background
        self._scale = possibly_create_parameter(scale, name="scale")
        self._bkg = possibly_create_parameter(bkg, name="bkg")

        # we can optimize the resolution (but this is always overridden by
        # x_err if supplied. There is therefore possibly no dependence on it.
        self._dq = possibly_create_parameter(dq, name="dq - resolution")
        self.dq_type = dq_type

        self._q_offset = possibly_create_parameter(
            q_offset, name="q_offset", units="Å**-1"
        )

        self._structure = None
        self.structure = structure

    def __call__(self, x, p=None, x_err=None):
        r"""
        Calculate the generative model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation.
            Units = Angstrom**-1
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
        return (
            f"ReflectModel({self._structure!r}, name={self.name!r},"
            f" scale={self.scale!r}, bkg={self.bkg!r},"
            f" dq={self.dq!r}, threads={self.threads},"
            f" quad_order={self.quad_order!r}, dq_type={self.dq_type!r},"
            f" q_offset={self.q_offset!r})"
        )

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
    def q_offset(self):
        r"""
        :class:`refnx.analysis.Parameter` - compensates for any angular
        misalignment during an experiment.

        """
        return self._q_offset

    @q_offset.setter
    def q_offset(self, value):
        self._q_offset.value = value

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
            Units = Angstrom**-1
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
        if x_err is None or self.dq_type == "constant":
            # fallback to what this object was constructed with
            x_err = float(self.dq)

        return reflectivity(
            x + self.q_offset.value,
            self.structure.slabs()[..., :4],
            scale=self.scale.value,
            bkg=self.bkg.value,
            dq=x_err,
            threads=self.threads,
            quad_order=self.quad_order,
        )

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
        p = Parameters(name="instrument parameters")
        p.extend([self.scale, self.bkg, self.dq, self.q_offset])

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


def reflectivity(
    q, slabs, scale=1.0, bkg=0.0, dq=5.0, quad_order=17, threads=-1
):
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
           (PDF). `dq` will have the shape (qvals.shape, 2, M).  There are
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
        return abeles(q, slabs, scale=scale, bkg=bkg, threads=threads)
    elif isinstance(dq, numbers.Real):
        dq = float(dq)
        return (
            scale * _smeared_abeles_constant(q, slabs, dq, threads=threads)
        ) + bkg

    # point by point resolution smearing (each q point has different dq/q)
    if isinstance(dq, np.ndarray) and dq.size == q.size:
        dqvals_flat = dq.flatten()
        qvals_flat = q.flatten()

        # adaptive quadrature
        if quad_order == "ultimate":
            smeared_rvals = (
                scale
                * _smeared_abeles_adaptive(
                    qvals_flat, slabs, dqvals_flat, threads=threads
                )
                + bkg
            )
            return smeared_rvals.reshape(q.shape)
        # fixed order quadrature
        else:
            smeared_rvals = (
                scale
                * _smeared_abeles_pointwise(
                    qvals_flat,
                    slabs,
                    dqvals_flat,
                    quad_order=quad_order,
                    threads=threads,
                )
                + bkg
            )
            return np.reshape(smeared_rvals, q.shape)

    # resolution kernel smearing
    elif (
        isinstance(dq, np.ndarray)
        and dq.ndim == q.ndim + 2
        and dq.shape[0 : q.ndim] == q.shape
    ):

        qvals_for_res = dq[:, 0, :]
        # work out the reflectivity at the kernel evaluation points
        smeared_rvals = abeles(qvals_for_res, slabs, threads=threads)

        # multiply by probability
        smeared_rvals *= dq[:, 1, :]

        # now do simpson integration
        rvals = scipy.integrate.simps(smeared_rvals, x=dq[:, 0, :])

        return scale * rvals + bkg

    return None


@lru_cache(maxsize=128)
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
    return abeles(localq, w, threads=threads) * gauss


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
    warnings.simplefilter("ignore", Warning)
    for idx, val in enumerate(qvals):
        smeared_rvals[idx], err = scipy.integrate.quadrature(
            _smearkernel,
            -_INTLIMIT,
            _INTLIMIT,
            tol=2 * np.finfo(np.float64).eps,
            rtol=2 * np.finfo(np.float64).eps,
            args=(w, qvals[idx], dqvals[idx], threads),
        )

    warnings.resetwarnings()
    return smeared_rvals


def _smeared_abeles_pointwise(qvals, w, dqvals, quad_order=17, threads=-1):
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
    prefactor = 1.0 / np.sqrt(2 * np.pi)

    def gauss(x):
        return np.exp(-0.5 * x * x)

    gaussvals = prefactor * gauss(abscissa * _INTLIMIT)

    # integration between -3.5 and 3.5 sigma
    va = qvals - _INTLIMIT * dqvals / _FWHM
    vb = qvals + _INTLIMIT * dqvals / _FWHM

    va = va[:, np.newaxis]
    vb = vb[:, np.newaxis]

    qvals_for_res = (np.atleast_2d(abscissa) * (vb - va) + vb + va) / 2.0
    smeared_rvals = abeles(qvals_for_res, w, threads=threads)

    smeared_rvals = np.reshape(smeared_rvals, (qvals.size, abscissa.size))

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
        return abeles(q, w, threads=threads)

    resolution /= 100
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    def gauss(x, s):
        return 1.0 / s / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2 / s / s)

    lowq = np.min(q)
    highq = np.max(q)
    if lowq <= 0:
        lowq = 1e-6

    start = np.log10(lowq) - 6 * resolution / _FWHM
    finish = np.log10(highq * (1 + 6 * resolution / _FWHM))
    interpnum = np.round(
        np.abs(
            1
            * (np.abs(start - finish))
            / (1.7 * resolution / _FWHM / gaussgpoint)
        )
    )
    xtemp = _cached_linspace(start, finish, int(interpnum))
    xlin = np.power(10.0, xtemp)

    # resolution smear over [-4 sigma, 4 sigma]
    gauss_x = _cached_linspace(-1.7 * resolution, 1.7 * resolution, gaussnum)
    gauss_y = gauss(gauss_x, resolution / _FWHM)

    rvals = abeles(xlin, w, threads=threads)
    smeared_rvals = np.convolve(rvals, gauss_y, mode="same")
    smeared_rvals *= gauss_x[1] - gauss_x[0]

    # interpolator = InterpolatedUnivariateSpline(xlin, smeared_rvals)
    #
    # smeared_output = interpolator(q)

    tck = splrep(xlin, smeared_rvals)
    smeared_output = splev(q, tck)

    return smeared_output


@lru_cache(maxsize=128)
def _cached_linspace(start, stop, num):
    # calculates linspace for _smeared_abeles_constant
    # this deserves a cache because it's called a lot with
    # the same parameters
    return np.linspace(start, stop, num)


class MixedReflectModel:
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
    dq_type: {'pointwise', 'constant'}, optional
        Chooses whether pointwise or constant dQ/Q resolution smearing (see
        `dq` keyword) is used. To use pointwise smearing the `x_err` keyword
        provided to `Objective.model` method must be an array, otherwise the
        smearing falls back to 'constant'.
    q_offset: float or refnx.analysis.Parameter, optional
        Compensates for uncertainties in the angle at which the measurement is
        performed. A positive/negative `q_offset` corresponds to a situation
        where the measured q values (incident angle) may have been under/over
        estimated, and has the effect of shifting the calculated model to
        lower/higher effective q values.
    """

    def __init__(
        self,
        structures,
        scales=None,
        bkg=1e-7,
        name="",
        dq=5.0,
        threads=-1,
        quad_order=17,
        dq_type="pointwise",
        q_offset=0.0,
    ):
        self.name = name
        self._parameters = None
        self.threads = threads
        self.quad_order = quad_order

        # all reflectometry models need a scale factor and background. Set
        # them all to 1 by default.
        pscales = Parameters(name="scale factors")

        if scales is not None and len(structures) == len(scales):
            tscales = scales
        elif scales is not None and len(structures) != len(scales):
            raise ValueError(
                "You need to supply scale factor for each" " structure"
            )
        else:
            tscales = [1 / len(structures)] * len(structures)

        for scale in tscales:
            p_scale = possibly_create_parameter(scale, name="scale")
            pscales.append(p_scale)

        self._scales = pscales
        self._bkg = possibly_create_parameter(bkg, name="bkg")

        # we can optimize the resolution (but this is always overridden by
        # x_err if supplied. There is therefore possibly no dependence on it.
        self._dq = possibly_create_parameter(dq, name="dq - resolution")
        self.dq_type = dq_type

        self._q_offset = possibly_create_parameter(
            q_offset, name="q_offset", units="Å**-1"
        )

        self._structures = structures

    def __repr__(self):
        s = (
            f"MixedReflectModel({self._structures!r},"
            f" scales={self._scales!r}, bkg={self._bkg!r},"
            f" name={self.name!r}, dq={self._dq!r},"
            f" threads={self.threads!r}, quad_order={self.quad_order!r},"
            f" dq_type={self.dq_type!r}, q_offset={self.q_offset!r})"
        )
        return s

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
    def q_offset(self):
        r"""
        :class:`refnx.analysis.Parameter` - compensates for any angular
        misalignment during an experiment.

        """
        return self._q_offset

    @q_offset.setter
    def q_offset(self, value):
        self._q_offset.value = value

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
        if x_err is None or self.dq_type == "constant":
            # fallback to what this object was constructed with
            x_err = float(self.dq)

        scales = np.array(self.scales)

        y = np.zeros_like(x)

        for scale, structure in zip(scales, self.structures):
            y += reflectivity(
                x + self.q_offset.value,
                structure.slabs()[..., :4],
                scale=scale,
                dq=x_err,
                threads=self.threads,
                quad_order=self.quad_order,
            )

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
        p = Parameters(name="instrument parameters")
        p.extend([self.scales, self.bkg, self.dq, self.q_offset])

        self._parameters = Parameters(name=self.name)
        self._parameters.append([p])
        self._parameters.extend(
            [structure.parameters for structure in self._structures]
        )
        return self._parameters


class FresnelTransform(Transform):
    """
    Fresnel transform for data.

    Divides experimental signal by reflectivity from an infinitely
    sharp interface.

    Parameters
    ----------
    sld_fronting: float
        SLD of fronting medium
    sld_backing: float
        SLD of backing medium
    dq : float, array-like optional
        - `dq == 0`
           no resolution smearing is employed.
        - `dq` is a float
           a constant dQ/Q resolution smearing is employed.  For 5% resolution
           smearing supply 5.
        - `dq` is array-like
           the array contains the FWHM of a Gaussian approximated resolution
           kernel. Point by point resolution smearing is employed.  Use this
           option if dQ/Q varies across your dataset.

    Notes
    -----
    Using a null reflectivity system (`sld_fronting == sld_backing`) will lead
    to `ZeroDivisionError`.
    If point-by-point resolution smearing is employed then a unique transform
    must be created for each Objective. This is because the number of points in
    `dq` must be the same as `x.size` when `FresnelTransform.transform` is
    called.
    """

    def __init__(self, sld_fronting, sld_backing, dq=0):
        self.form = "Fresnel"

        self.sld_fronting = sld_fronting
        self.sld_backing = sld_backing
        self.dq = dq

    def __repr__(self):
        # use repr for sld's because they could be `reflect.SLD` objects
        return "FresnelTransform({0}, {1}, dq={2})".format(
            repr(self.sld_fronting), repr(self.sld_backing), repr(self.dq)
        )

    def __call__(self, x, y, y_err=None, x_err=0):
        """
        Calculate the transformed data

        Parameters
        ----------
        x : array-like
            x-values
        y : array-like
            y-values
        y_err : array-like
            Uncertainties in `y` (standard deviation)
        x_err : array-like
            Uncertainties in `x` (*FWHM*)

        Returns
        -------
        yt, et : tuple
            The transformed data

        """
        return self.__transform(x, y, y_err=y_err)

    def __transform(self, x, y, y_err=None):
        r"""
        Transform the data passed in

        Parameters
        ----------
        x : array-like
            x-values
        y : array-like
            y-values
        y_err : array-like
            Uncertainties in `y` (standard deviation)
        x_err : array-like
            Uncertainties in `x` (*FWHM*)

        Returns
        -------
        yt, et : tuple
            The transformed data
        """
        sld_fronting = complex(self.sld_fronting)
        sld_backing = complex(self.sld_backing)

        slabs = np.array(
            [[0, sld_fronting.real, 0, 0], [0, sld_backing.real, 0, 0]]
        )

        fresnel = reflectivity(x, slabs, dq=self.dq)

        yt = np.copy(y)
        yt /= fresnel

        if y_err is None:
            return yt, None
        else:
            return yt, y_err / fresnel


def choose_dq_type(objective):
    """
    Chooses which resolution smearing approach has the
    fastest calculation time.

    Parameters
    ----------
    objective: Objective
        The objective being calculated

    Returns
    -------
    method: str
        One of {'pointwise', 'constant'}. If 'pointwise' then using
        the resolution information from the datafile is the fastest mode
        of calculation. If 'constant', then a constant dq/q (expressed as
        a percentage) Q resolution is quicker.
    """
    # choose which resolution smearing approach to use
    if objective.data.x_err is None or not isinstance(
        objective.model, (ReflectModel, MixedReflectModel)
    ):
        return "constant"

    original_method = objective.model.dq_type

    # time how long point-by-point takes
    objective.model.dq_type = "pointwise"
    start = time.time()
    for i in range(100):
        objective.generative()
    time_pp = time.time() - start

    x_err = objective.data.x_err
    objective.data.x_err = None
    dq = 10.0 * x_err / objective.data.x
    objective.model.dq.value = np.mean(dq)
    objective.model.dq_type = "constant"
    start = time.time()
    for i in range(100):
        objective.generative()
    const_pp = time.time() - start

    # replace original state
    objective.data.x_err = x_err
    objective.model.dq_type = original_method
    #     print(f"Constant: {const_pp}, point-by-point: {time_pp}")
    if const_pp < time_pp:
        # if constant resolution smearing better.
        return "constant"
    return "pointwise"
