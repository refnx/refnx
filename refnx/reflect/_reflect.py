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

from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import Optional

# TINY = np.finfo(np.float64).tiny
TINY = 1e-30

"""
import numpy as np
q = np.linspace(0.01, 0.5, 1000)
w = np.array([[0, 2.07, 0, 0],
              [100, 3.47, 0, 3],
              [500, -0.5, 0.00001, 3],
              [0, 6.36, 0, 3]])
"""

"""
The timings for the reflectivity calculation above are (6/3/2019):

_creflect.abeles = 254 us
_reflect.abeles = 433 us
the alternative cython implementation is 572 us.

If TINY is made too small, then the C implementations start too suffer because
the sqrt calculation takes too long. The C implementation is only just ahead of
the python implementation!
"""


class _Abeles_pyopencl:
    def __init__(self):
        self.ctx = None
        self.prg = None

    def __getstate__(self):
        # pyopencl Contexts and Programs can't be pickled.
        d = self.__dict__
        d["ctx"] = None
        d["prg"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __call__(self, q, w, scale=1.0, bkg=0.0, threads=0):
        """
        Abeles matrix formalism for calculating reflectivity from a
        stratified
        medium.
        Uses pyopencl on a GPU to calculate reflectivity. The accuracy of
        this function may not as good as the C and Python based versions.
        Furthermore, it can be tricky to use when using multiprocessing
        based parallelism.

        Parameters
        ----------
        q: array_like
            the q values required for the calculation.
            Q = 4 * Pi / lambda * sin(omega).
            Units = Angstrom**-1
        layers: np.ndarray
            coefficients required for the calculation, has shape
            (2 + N, 4), where N is the number of layers
            layers[0, 1] - SLD of fronting (/1e-6 Angstrom**-2)
            layers[0, 2] - iSLD of fronting (/1e-6 Angstrom**-2)
            layers[N, 0] - thickness of layer N
            layers[N, 1] - SLD of layer N (/1e-6 Angstrom**-2)
            layers[N, 2] - iSLD of layer N (/1e-6 Angstrom**-2)
            layers[N, 3] - roughness between layer N-1/N
            layers[-1, 1] - SLD of backing (/1e-6 Angstrom**-2)
            layers[-1, 2] - iSLD of backing (/1e-6 Angstrom**-2)
            layers[-1, 3] - roughness between backing and last layer
        scale: float
            Multiply all reflectivities by this value.
        bkg: float
            Linear background to be added to all reflectivities
        threads: int, optional
            <THIS OPTION IS CURRENTLY IGNORED>

        Returns
        -------
        Reflectivity: np.ndarray
            Calculated reflectivity values for each q value.
        """
        import pyopencl as cl

        if self.ctx is None or self.prg is None:
            self.ctx = cl.create_some_context(interactive=False)
            pth = Path(__file__).absolute().parent
            with open(pth / "abeles_pyopencl.cl", "r") as f:
                src = f.read()
            self.prg = cl.Program(self.ctx, src).build()

        qvals = np.asarray(q).astype(float)
        flatq = qvals.ravel()

        nlayers = len(w) - 2
        coefs = np.empty((nlayers * 4 + 8))
        coefs[0] = nlayers
        coefs[1] = scale
        coefs[2:4] = w[0, 1:3]
        coefs[4:6] = w[-1, 1:3]
        coefs[6] = bkg
        coefs[7] = w[-1, 3]
        if nlayers:
            coefs[8::4] = w[1:-1, 0]
            coefs[9::4] = w[1:-1, 1]
            coefs[10::4] = w[1:-1, 2]
            coefs[11::4] = w[1:-1, 3]

        mf = cl.mem_flags
        with cl.CommandQueue(self.ctx) as queue:
            q_g = cl.Buffer(
                self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flatq
            )
            coefs_g = cl.Buffer(
                self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=coefs
            )
            ref_g = cl.Buffer(self.ctx, mf.WRITE_ONLY, flatq.nbytes)

            self.prg.abeles(queue, flatq.shape, None, q_g, coefs_g, ref_g)

            reflectivity = np.empty_like(flatq)
            cl.enqueue_copy(queue, reflectivity, ref_g)
        return np.reshape(reflectivity, qvals.shape)


abeles_pyopencl = _Abeles_pyopencl()


def abeles(
    q: npt.ArrayLike,
    layers: npt.ArrayLike,
    scale: Optional[float] = 1.0,
    bkg: Optional[float] = 0,
    threads: Optional[int] = 0,
) -> np.array:
    """
    Abeles matrix formalism for calculating reflectivity from a stratified
    medium.

    Parameters
    ----------
    q: array_like
        the q values required for the calculation.
        Q = 4 * Pi / lambda * sin(omega).
        Units = Angstrom**-1
    layers: np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers
        layers[0, 1] - SLD of fronting (/1e-6 Angstrom**-2)
        layers[0, 2] - iSLD of fronting (/1e-6 Angstrom**-2)
        layers[N, 0] - thickness of layer N
        layers[N, 1] - SLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 2] - iSLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 3] - roughness between layer N-1/N
        layers[-1, 1] - SLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 2] - iSLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 3] - roughness between backing and last layer
    scale: float
        Multiply all reflectivities by this value.
    bkg: float
        Linear background to be added to all reflectivities
    threads: int, optional
        <THIS OPTION IS CURRENTLY IGNORED>

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """
    qvals = np.asarray(q).astype(float, copy=False)
    flatq = qvals.ravel()

    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    kn = np.zeros((npnts, nlayers + 2), np.complex128)
    mi00 = np.ones((npnts, nlayers + 1), np.complex128)

    sld = np.zeros(nlayers + 2, np.complex128)

    # addition of TINY is to ensure the correct branch cut
    # in the complex sqrt calculation of kn.
    sld[1:] += (
        (layers[1:, 1] - layers[0, 1]) + 1j * (np.abs(layers[1:, 2]) + TINY)
    ) * 1.0e-6

    # kn is a 2D array. Rows are Q points, columns are kn in a layer.
    # calculate wavevector in each layer, for each Q point.
    kn[:] = np.sqrt(flatq[:, np.newaxis] ** 2.0 / 4.0 - 4.0 * np.pi * sld)

    # reflectances for each layer
    # rj.shape = (npnts, nlayers + 1)
    rj = kn[:, :-1] - kn[:, 1:]
    rj /= kn[:, :-1] + kn[:, 1:]
    rj *= np.exp(-2.0 * kn[:, :-1] * kn[:, 1:] * layers[1:, 3] ** 2)

    # characteristic matrices for each layer
    # miNN.shape = (npnts, nlayers + 1)
    if nlayers:
        mi00[:, 1:] = np.exp(kn[:, 1:-1] * 1j * np.fabs(layers[1:-1, 0]))
    mi11 = 1.0 / mi00
    mi10 = rj * mi00
    mi01 = rj * mi11

    # initialise matrix total
    mrtot00 = mi00[:, 0]
    mrtot01 = mi01[:, 0]
    mrtot10 = mi10[:, 0]
    mrtot11 = mi11[:, 0]

    # propagate characteristic matrices
    for idx in range(1, nlayers + 1):
        # matrix multiply mrtot by characteristic matrix
        p0 = mrtot00 * mi00[:, idx] + mrtot10 * mi01[:, idx]
        p1 = mrtot00 * mi10[:, idx] + mrtot10 * mi11[:, idx]
        mrtot00 = p0
        mrtot10 = p1

        p0 = mrtot01 * mi00[:, idx] + mrtot11 * mi01[:, idx]
        p1 = mrtot01 * mi10[:, idx] + mrtot11 * mi11[:, idx]

        mrtot01 = p0
        mrtot11 = p1

    r = mrtot01 / mrtot00
    reflectivity = r * np.conj(r)
    reflectivity *= scale
    reflectivity += bkg
    return np.real(np.reshape(reflectivity, qvals.shape))


def parratt(
    q,
    layers,
    scale=1.0,
    bkg=0,
    threads=0,
) -> np.array:
    """
    Parratt recursion formula for calculating reflectivity from a stratified
    medium.

    Parameters
    ----------
    q: array_like
        the q values required for the calculation.
        Q = 4 * Pi / lambda * sin(omega).
        Units = Angstrom**-1
    layers: np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers
        layers[0, 1] - SLD of fronting (/1e-6 Angstrom**-2)
        layers[0, 2] - iSLD of fronting (/1e-6 Angstrom**-2)
        layers[N, 0] - thickness of layer N
        layers[N, 1] - SLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 2] - iSLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 3] - roughness between layer N-1/N
        layers[-1, 1] - SLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 2] - iSLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 3] - roughness between backing and last layer
    scale: float
        Multiply all reflectivities by this value.
    bkg: float
        Linear background to be added to all reflectivities
    threads: int, optional
        <THIS OPTION IS CURRENTLY IGNORED>

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """
    qvals = np.asarray(q).astype(float, copy=False)
    flatq = qvals.ravel()

    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    kn = np.zeros((npnts, nlayers + 2), np.complex128)
    sld = np.zeros(nlayers + 2, np.complex128)

    # addition of TINY is to ensure the correct branch cut
    # in the complex sqrt calculation of kn.
    sld[1:] += (
        (layers[1:, 1] - layers[0, 1]) + 1j * (np.abs(layers[1:, 2]) + TINY)
    ) * 1.0e-6

    # calculate wavevector in each layer, for each Q point.
    # kn.shape = (npnts, nlayers)
    kn[:] = np.sqrt(flatq[:, np.newaxis] ** 2.0 / 4.0 - 4.0 * np.pi * sld)

    # reflectances for each layer
    # rj.shape = (npnts, nlayers + 1)
    rj = kn[:, :-1] - kn[:, 1:]
    rj /= kn[:, :-1] + kn[:, 1:]
    rj *= np.exp(-2.0 * kn[:, :-1] * kn[:, 1:] * layers[1:, 3] ** 2)

    beta = np.exp(
        -2.0
        * kn[:, 1 : nlayers + 1]
        * 1j
        * np.fabs(layers[1 : nlayers + 1, 0])
    )
    beta_rj = beta * rj[:, 0:nlayers]

    RRJ_1 = rj[:, -1]
    for idx in range(nlayers - 1, -1, -1):
        # RRJ = (rj[:, idx] + RRJ_1 * beta[:, idx]) / (1 + rj[:, idx] * RRJ_1 * beta[:, idx])
        RRJ = (rj[:, idx] + RRJ_1 * beta[:, idx]) / (
            1 + RRJ_1 * beta_rj[:, idx]
        )
        RRJ_1 = RRJ

    reflectivity = RRJ_1 * np.conj(RRJ_1)
    reflectivity *= scale
    reflectivity += bkg
    return np.real(np.reshape(reflectivity, qvals.shape))


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
    The reflectivity profiles from both contracted and un-contracted profiles
    should be compared to check for accuracy.
    """

    # In refl1d the first slab is the substrate, the order is reversed here.
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
        dz = rhoarea = irhoarea = vfsolvarea = 0.0
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
            if i == n or sigma[i - 1] != 0.0:
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


"""
Polarised Neutron Reflectometry calculation
"""


def _pmatrix(kn_u, kn_d, thickness):
    """
    # equation 7 + 14 in Blundell and Bland

    Parameters
    ----------
    kn_u, kn_d: np.ndarray
        wavevector for up and down within a given layer. Has shape (N,),
        where N is the number of Q points.

    thickness: float
        Thickness of layer (Angstrom)

    Returns
    -------
    p : np.ndarray
        P matrix
    """
    p = np.zeros((kn_u.size, 4, 4), np.complex128)

    p0 = np.exp(complex(0, 1) * kn_u * thickness)
    p1 = np.exp(complex(0, 1) * kn_d * thickness)

    p[:, 0, 0] = 1 / p0
    p[:, 1, 1] = p0
    p[:, 2, 2] = 1 / p1
    p[:, 3, 3] = p1

    return p


def _dmatrix(kn_u, kn_d):
    """
    equation 5 + 13 in Blundell and Bland

    Parameters
    ----------
    kn_u, kn_d: np.ndarray
        wavevector for up and down within a given layer. Has shape (N,),
        where N is the number of Q points.

    Returns
    -------
    d, d_inv: np.ndarray
        D matrix and its inverse
    """
    d = np.zeros((kn_u.size, 4, 4), np.complex128)
    d_inv = np.zeros_like(d)

    d[:, 0, 0] = 1
    d[:, 0, 1] = 1
    d[:, 1, 0] = kn_u
    d[:, 1, 1] = -kn_u

    d[:, 2, 2] = 1
    d[:, 2, 3] = 1
    d[:, 3, 2] = kn_d
    d[:, 3, 3] = -kn_d

    # an analytic matrix inverse saves time
    inv_kn_u = 0.5 / kn_u
    inv_kn_d = 0.5 / kn_d

    d_inv[:, 0, 0] = 0.5
    d_inv[:, 0, 1] = inv_kn_u
    d_inv[:, 1, 0] = 0.5
    d_inv[:, 1, 1] = -inv_kn_u

    d_inv[:, 2, 2] = 0.5
    d_inv[:, 2, 3] = inv_kn_d
    d_inv[:, 3, 2] = 0.5
    d_inv[:, 3, 3] = -inv_kn_d

    return d, d_inv


def _rmatrix(theta):
    """
    equation 15 in Blundell and Bland

    Parameters
    ----------
    theta - float
        Angle (degrees) of magnetic moment with respect to applied field.

    Returns
    -------
    r : np.ndarray
        R matrix.
    """
    r = np.zeros((4, 4), np.complex128)

    cos_term = np.cos(theta / 2.0) * complex(1, 0)
    sin_term = np.sin(theta / 2.0) * complex(1, 0)

    r[0, 0] = cos_term
    r[1, 1] = cos_term

    r[0, 2] = sin_term
    r[1, 3] = sin_term

    r[2, 0] = -sin_term
    r[3, 1] = -sin_term

    r[2, 2] = cos_term
    r[3, 3] = cos_term

    return r


def _magsqr(z):
    """
    Return the magnitude squared of the real- or complex-valued input.

    Parameters
    ----------
    z - complex, or np.ndarray
        complex argument

    Returns
    -------
    magsqr - real or np.ndarray
        Magnitude squared of the complex argument
    """
    return np.abs(z) ** 2


def pnr(q, layers):
    """
    Calculates Polarised Neutron Reflectivity of a series of slabs.

    No interlayer roughness is taken into account.

    Parameters
    ----------
    q: array_like
        the q values required for the calculation.
        Q = 4 * Pi / lambda * sin(omega).
        Units = Angstrom**-1
    layers: np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers
        layers[0, 1] - SLD of fronting (/1e-6 Angstrom**-2)
        layers[0, 2] - iSLD of fronting (/1e-6 Angstrom**-2)
        layers[0, 3] - magSLD of fronting (/1e-6 Angstrom**-2)
        layers[0, 3] - angle of magnetic moment w.r.t applied field (degrees)

        layers[N, 0] - thickness of layer N
        layers[N, 1] - SLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 2] - iSLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 3] - magSLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 4] - angle of magnetic moment w.r.t applied field (degrees)

        layers[-1, 1] - SLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 2] - iSLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 3] - magSLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 4] - angle of magnetic moment w.r.t applied field (degrees)

    Returns
    -------
    reflectivity: tuple of np.ndarray
        Calculated Polarised Neutron Reflectivity values for each q value.
        (PP, MM, PM, MP)

    References
    ----------
    ..[1] S. J. Blundell, J. A. C. Bland, 'Polarized neutron reflection as a
         probe of magnetic films and multilayers', Phys. Rev. B, (1992), 46,
         3391.
    """
    xx = np.asarray(q).astype(np.complex128).ravel()

    thetas = np.radians(layers[:, 4])
    thetas = np.diff(thetas)

    # nuclear SLD minus that of the superphase
    sld = layers[:, 1] + 1j * layers[:, 2] - layers[0, 1] - 1j * layers[0, 2]

    # nuclear and magnetic
    sldu = sld + layers[:, 3] - layers[0, 3]
    sldd = sld - layers[:, 3] + layers[0, 3]
    sldu *= 1e-6
    sldd *= 1e-6

    # wavevector in each layer
    kn_u = np.sqrt(0.25 * xx[:, np.newaxis] ** 2 - 4 * np.pi * sldu)
    kn_d = np.sqrt(0.25 * xx[:, np.newaxis] ** 2 - 4 * np.pi * sldd)

    mm = np.zeros((xx.size, 4, 4), np.complex128)
    mm[:] = np.identity(4, np.complex128)

    # iterate over layers
    for jj in range(len(layers) - 2):
        d, d_inv = _dmatrix(kn_u[:, jj + 1], kn_d[:, jj + 1])
        p = _pmatrix(kn_u[:, jj + 1], kn_d[:, jj + 1], layers[jj + 1, 0])
        r = _rmatrix(thetas[jj + 1])

        mm = mm @ d @ p @ d_inv @ r

    # d_inv for the first layer
    _, d_inv = _dmatrix(kn_u[:, 0], kn_d[:, 0])

    # d for the last layer
    d, _ = _dmatrix(kn_u[:, -1], kn_d[:, -1])
    r = _rmatrix(thetas[0])

    M = d_inv @ r @ mm @ d

    # equation 16 in Blundell and Bland
    den = M[:, 0, 0] * M[:, 2, 2] - M[:, 0, 2] * M[:, 2, 0]
    # uu
    pp = _magsqr((M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0]) / den)

    # dd
    mm = _magsqr((M[:, 3, 2] * M[:, 0, 0] - M[:, 3, 0] * M[:, 0, 2]) / den)

    # ud
    pm = _magsqr((M[:, 3, 0] * M[:, 2, 2] - M[:, 3, 2] * M[:, 2, 0]) / den)

    # du
    mp = _magsqr((M[:, 1, 2] * M[:, 0, 0] - M[:, 1, 0] * M[:, 0, 2]) / den)

    return (pp, mm, pm, mp)


# w is a 2D array with each row being [d, real SLD, imag SLD, roughness, moment, thetaM], where thetaM is angle
# of applied field and moment.
# def bb(q, w):
#     qvals = np.asarray(q).astype(float, copy=False)
#     flatq = qvals.ravel()
#
#     nlayers = w.shape[0] - 2
#     npnts = flatq.size
#
#     k_u = np.zeros((npnts, nlayers + 2), np.complex128)
#     k_d = np.zeros((npnts, nlayers + 2), np.complex128)
#     sld_u = np.zeros(nlayers + 2, np.complex128)
#     sld_d = np.zeros(nlayers + 2, np.complex128)
#
#     thick = w[1:-1, 0]
#     rough2 = (-2.0 * w[1:, 3] ** 2)[:, None]  # (nlayers + 1, 1)
#     mag = w[:, 4]
#     angle = np.radians(w[:, 5])
#     beta = np.ediff1d(angle)
#
#     # addition of TINY is to ensure the correct branch cut
#     # in the complex sqrt calculation of kn.
#     # TODO check what moment is supposed to be in front layer
#     # sld_u[1:] += (
#     #     (w[1:, 1] + mag[1:] - (w[0, 1] + mag[0]))
#     #     + 1j * (np.abs(w[1:, 2]) + TINY)
#     # ) * 1.0e-6
#
#     # sld_d[1:] += (
#     #     (w[1:, 1] - mag[1:] - (w[0, 1] - mag[0]))
#     #     + 1j * (np.abs(w[1:, 2]) + TINY)
#     # ) * 1.0e-6
#
#     sld_u[1:] += ((w[1:, 1] + mag[1:]) + 1j * (np.abs(w[1:, 2]))) * 1.0e-6
#     sld_d[1:] += ((w[1:, 1] - mag[1:]) + 1j * (np.abs(w[1:, 2]))) * 1.0e-6
#
#     k0 = (q / 2)[:, None]
#
#     k_u[:] = k0 * np.sqrt(1 - (4 * np.pi * sld_u[None, :]) / k0 ** 2)
#     k_d[:] = k0 * np.sqrt(1 - (4 * np.pi * sld_d[None, :]) / k0 ** 2)
#
#     # wavevectors in each of the layers (npnts, nlayers + 2)
#     # k_u[:] = np.sqrt(flatq[:, np.newaxis] ** 2.0 / 4.0 - 4.0 * np.pi * sld_u)
#     # k_d[:] = np.sqrt(flatq[:, np.newaxis] ** 2.0 / 4.0 - 4.0 * np.pi * sld_d)
#
#     # (nlayers + 2, npnts)
#     k_u = k_u.T
#     k_d = k_d.T
#
#     ########################################
#     # forward values, i.e. r_{m-1, m, r_{12}
#
#     # (nlayers + 1, 1)
#     beta_2 = (beta / 2.0)[:, None]
#     sin_beta_2 = np.sin(beta_2)
#     cos_beta_2 = np.cos(beta_2)
#     sin2_beta_2 = sin_beta_2 ** 2
#     cos2_beta_2 = cos_beta_2 ** 2
#
#     den = (k_u[:-1] + k_u[1:]) * (k_d[:-1] + k_d[1:]) * cos2_beta_2
#     den += (k_u[:-1] + k_d[1:]) * (k_d[:-1] + k_u[1:]) * sin2_beta_2
#
#     # each of the reflectances/transmittances are (nlayers + 1, npnts)
#     rpp = (
#                   cos2_beta_2 * (k_u[:-1] - k_u[1:]) * (k_d[:-1] + k_d[1:])
#                   + sin2_beta_2 * (k_u[:-1] - k_d[1:]) * (k_d[:-1] + k_u[1:])
#           ) / den
#     rpm = 2 * k_u[:-1] * sin_beta_2 * cos_beta_2 * (k_u[1:] - k_d[1:]) / den
#     tpp = 2 * cos_beta_2 * k_u[:-1] * (k_d[:-1] + k_d[1:]) / den
#     tpm = 2 * sin_beta_2 * k_u[:-1] * (k_d[:-1] + k_u[1:]) / den
#
#     # symmetry
#     rmm = (
#                   cos2_beta_2 * (k_d[:-1] - k_d[1:]) * (k_u[:-1] + k_u[1:])
#                   + sin2_beta_2 * (k_d[:-1] - k_u[1:]) * (k_u[:-1] + k_d[1:])
#           ) / den
#     rmp = 2 * sin_beta_2 * cos_beta_2 * k_d[:-1] * (k_d[1:] - k_u[1:]) / den
#     tmm = 2 * cos_beta_2 * k_d[:-1] * (k_u[:-1] + k_u[1:]) / den
#     tmp = 2 * sin_beta_2 * k_d[:-1] * (k_u[:-1] + k_d[1:]) / den
#
#     # modify reflectance by Debye-Waller factor
#     # rough2 already incorporates a factor of -2.0.
#     rpp *= np.exp(k_u[:-1] * k_u[1:] * rough2)
#     rpm *= np.exp(k_u[:-1] * k_d[1:] * rough2)
#     rmm *= np.exp(k_d[:-1] * k_d[1:] * rough2)
#     rmp *= np.exp(k_d[:-1] * k_u[1:] * rough2)
#
#     ########################################
#     # backward values, i.e. r_{m, m-1}, r_{21}
#     # each of the reflectances/transmittances are (nlayers + 1, npnts)
#     beta_2 = (-beta / 2.0)[:, None]
#     sin_beta_2 = np.sin(beta_2)
#     cos_beta_2 = np.cos(beta_2)
#     sin2_beta_2 = sin_beta_2 ** 2
#     cos2_beta_2 = cos_beta_2 ** 2
#
#     den = (k_u[:-1] + k_u[1:]) * (k_d[:-1] + k_d[1:]) * cos2_beta_2
#     den += (k_u[:-1] + k_d[1:]) * (k_d[:-1] + k_u[1:]) * sin2_beta_2
#
#     rpp_back = (
#                        cos2_beta_2 * (k_u[1:] - k_u[:-1]) * (k_d[1:] + k_d[:-1])
#                        + sin2_beta_2 * (k_u[1:] - k_d[:-1]) * (k_d[1:] + k_u[:-1])
#                ) / den
#     rpm_back = 2 * sin_beta_2 * cos_beta_2 * k_u[1:] * (k_u[:-1] - k_d[:-1]) / den
#     tpp_back = 2 * cos_beta_2 * k_u[1:] * (k_d[:-1] + k_d[1:]) / den
#     tpm_back = 2 * sin_beta_2 * k_u[1:] * (k_d[1:] + k_u[:-1]) / den
#
#     # symmetry
#     rmm_back = (
#                        cos2_beta_2 * (k_d[1:] - k_d[:-1]) * (k_u[1:] + k_u[:-1])
#                        + sin2_beta_2 * (k_d[1:] - k_u[:-1]) * (k_u[1:] + k_d[:-1])
#                ) / den
#     rmp_back = 2 * sin_beta_2 * cos_beta_2 * k_d[1:] * (k_d[:-1] - k_u[:-1]) / den
#     tmm_back = 2 * cos_beta_2 * k_d[1:] * (k_u[:-1] + k_u[1:]) / den
#     tmp_back = 2 * sin_beta_2 * k_d[1:] * (k_u[1:] + k_d[:-1]) / den
#
#     # modify reflectance by Debye-Waller factor
#     # rough2 already incorporates a factor of -2.0.
#     rpp_back *= np.exp(k_u[:-1] * k_u[1:] * rough2)
#     rpm_back *= np.exp(k_u[1:] * k_d[:-1] * rough2)
#     rmm_back *= np.exp(k_d[:-1] * k_d[1:] * rough2)
#     rmp_back *= np.exp(k_d[1:] * k_u[:-1] * rough2)
#
#     # k.shape == (nlayers + 2, npnts), thick.shape == (nlayers)
#     # phi.shape == (nlayers, npnts)
#     if nlayers:
#         phi_u = np.exp(1J * k_u[1:-1] * thick)
#         phi_d = np.exp(1J * k_d[1:-1] * thick)
#
#     # eqn 9.79. Note that we're changing axes around to make it easier to matrix multiply further on
#     rij1 = np.zeros((npnts, 2, 2), dtype=np.complex128)
#     rij1[:, 0, 0] = rpp[-1]
#     rij1[:, 0, 1] = rpm[-1]
#     rij1[:, 1, 0] = rmp[-1]
#     rij1[:, 1, 1] = rmm[-1]
#
#     rij = np.zeros_like(rij1)
#     rij_back = np.zeros_like(rij1)
#     tij = np.zeros_like(rij1)
#     tij_back = np.zeros_like(rij1)
#     P = np.zeros_like(rij1)
#     EYE = np.eye(2, dtype=np.complex128)[None, ...]
#
#     # now work forward and multiply everything out
#     for idx in range(nlayers - 1, -1, -1):
#         # TODO look into a quicker way of assembling these
#         rij[:, 0, 0] = rpp[idx]
#         rij[:, 0, 1] = rpm[idx]
#         rij[:, 1, 0] = rmp[idx]
#         rij[:, 1, 1] = rmm[idx]
#
#         rij_back[:, 0, 0] = rpp_back[idx]
#         rij_back[:, 0, 1] = rpm_back[idx]
#         rij_back[:, 1, 0] = rmp_back[idx]
#         rij_back[:, 1, 1] = rmm_back[idx]
#
#         tij[:, 0, 0] = tpp[idx]
#         tij[:, 0, 1] = tpm[idx]
#         tij[:, 1, 0] = tmp[idx]
#         tij[:, 1, 1] = tmm[idx]
#
#         tij_back[:, 0, 0] = tpp_back[idx]
#         tij_back[:, 0, 1] = tpm_back[idx]
#         tij_back[:, 1, 0] = tmp_back[idx]
#         tij_back[:, 1, 1] = tmm_back[idx]
#
#         P[:, 0, 0] = phi_u[idx]
#         P[:, 1, 1] = phi_d[idx]
#
#         P_rij1_P = P @ rij1 @ P
#         rij1 = rij + tij @ P_rij1_P @ np.linalg.inv(EYE - rij_back @ P_rij1_P) @ tij_back
#
#     return rij1[:, 0, 0], rij1[:, 0, 1], rij1[:, 1, 0], rij1[:, 1, 1]


if __name__ == "__main__":
    a = np.zeros(12)
    a[0] = 1.0
    a[1] = 1.0
    a[4] = 2.07
    a[7] = 3
    a[8] = 100
    a[9] = 3.47
    a[11] = 2

    b = np.arange(1000.0)
    b /= 2000.0
    b += 0.001

    def loop():
        abeles(b, a)

    for i in range(1000):
        loop()
