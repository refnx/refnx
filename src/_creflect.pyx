# cython: language_level=3, boundscheck=False
"""
    *Calculates the specular (Neutron or X-ray) reflectivity from a stratified
    series of layers.

The refnx code is distributed under the following license:

Copyright (c) 2015 A. R. J. Nelson, Australian Nuclear Science and Technology Organisation

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
from multiprocessing import cpu_count
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport libcpp.complex
import numpy as np
cimport numpy as np
cimport cython


np.import_array()

ctypedef libcpp.complex.complex[double] Cplx
ctypedef np.float64_t float64_t
ctypedef np.complex128_t complex128_t


cdef extern from "refcaller.h" nogil:
    void abeles_wrapper(
        int numcoefs,
        const double *coefP,
        int npoints,
        double *yP,
        const double *xP
    )
    void abeles_wrapper_MT(
        unsigned int batch,
        int numcoefs,
        const double *coefP,
        int npoints,
        double *yP,
        const double *xP,
        int threads
    )
    void parratt_wrapper(
        int numcoefs,
        const double *coefP,
        int npoints,
        double *yP,
        const double *xP
    )
    void parratt_wrapper_MT(
        unsigned int batch,
        int numcoefs,
        const double *coefP,
        int npoints,
        double *yP,
        const double *xP,
        int threads
    )


cdef extern from "pnr/reflcalc.h" nogil:
    void calculate_U1_U3(
        const double H,
        double &rhoM,
        const double thetaM,
        const double Aguide,
        Cplx &U1,
        Cplx &U3
    )

    void magnetic_amplitude(
        const int layers,
        const double d[],
        const double sigma[],
        const double rho[],
        const double irho[],
        const double rhoM[],
        const Cplx u1[],
        const Cplx u3[],
        const int points,
        const double kz[],
        const int rho_offset[],
        Cplx Ra[],
        Cplx Rb[],
        Cplx Rc[],
        Cplx Rd[]
    )


# figure out CPU count
cdef int NCPU = cpu_count()


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray abeles(
    np.ndarray x,
    double[:, :] w,
    double scale=1.0,
    double bkg=0.,
    int threads=-1
):
    """Abeles matrix formalism for calculating reflectivity from a stratified
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
        How many threads you would like to use in the reflectivity calculation.
        If `threads == -1` then the calculation is automatically spread over
        `multiprocessing.cpu_count()` threads.

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """
    if w.shape[1] != 4 or w.shape[0] < 2:
        raise ValueError("Layer parameters for _creflect must be an array of"
                         " shape (>2, 4)")
    if x.dtype != np.float64:
        raise ValueError("Q values for _creflect must be np.float64")

    cdef:
        int nlayers = w.shape[0] - 2
        int npoints = x.size
        np.ndarray y = np.empty_like(x, np.float64)
        double *x_data
        double *y_data
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x, dtype=np.float64)

    x_data = <float64_t *>np.PyArray_DATA(x)
    y_data = <float64_t *>np.PyArray_DATA(y)

    coefs = <double*> PyMem_Malloc((4*nlayers + 8) * sizeof(double))
    if not coefs:
        raise MemoryError()

    cdef double [:] coefs_view = <double[:4*nlayers + 8]>coefs

    try:
        with nogil:
            if threads == -1:
                threads = NCPU
            elif threads == 0:
                threads = 1

            coefs_view[0] = nlayers
            coefs_view[1] = scale
            coefs_view[2:4] = w[0, 1: 3]
            coefs_view[4: 6] = w[-1, 1: 3]
            coefs_view[6] = bkg
            coefs_view[7] = w[-1, 3]

            if nlayers:
                coefs_view[8::4] = w[1:-1, 0]
                coefs_view[9::4] = w[1:-1, 1]
                coefs_view[10::4] = w[1:-1, 2]
                coefs_view[11::4] = w[1:-1, 3]

            if threads > 1:
                abeles_wrapper_MT(
                    1,
                    4*nlayers + 8,
                    coefs,
                    npoints,
                    y_data,
                    x_data,
                    threads
                )
            else:
                abeles_wrapper(
                    4*nlayers + 8,
                    coefs,
                    npoints,
                    y_data,
                    x_data
                )
    finally:
        PyMem_Free(coefs)

    return y


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray parratt(
    np.ndarray x,
    double[:, :] w,
    double scale=1.0,
    double bkg=0.,
    int threads=-1
):
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
        How many threads you would like to use in the reflectivity calculation.
        If `threads == -1` then the calculation is automatically spread over
        `multiprocessing.cpu_count()` threads.

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """
    if w.shape[1] != 4 or w.shape[0] < 2:
        raise ValueError("Layer parameters for _creflect must be an array of"
                         " shape (>2, 4)")
    if x.dtype != np.float64:
        raise ValueError("Q values for _creflect must be np.float64")

    cdef:
        int nlayers = w.shape[0] - 2
        int npoints = x.size
        np.ndarray y = np.empty_like(x, np.float64)
        double *x_data
        double *y_data
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x, dtype=np.float64)

    x_data = <float64_t *>np.PyArray_DATA(x)
    y_data = <float64_t *>np.PyArray_DATA(y)

    coefs = <double*> PyMem_Malloc((4*nlayers + 8) * sizeof(double))
    if not coefs:
        raise MemoryError()

    cdef double [:] coefs_view = <double[:4*nlayers + 8]>coefs

    try:
        with nogil:
            if threads == -1:
                threads = NCPU
            elif threads == 0:
                threads = 1

            coefs_view[0] = nlayers
            coefs_view[1] = scale
            coefs_view[2:4] = w[0, 1: 3]
            coefs_view[4: 6] = w[-1, 1: 3]
            coefs_view[6] = bkg
            coefs_view[7] = w[-1, 3]

            if nlayers:
                coefs_view[8::4] = w[1:-1, 0]
                coefs_view[9::4] = w[1:-1, 1]
                coefs_view[10::4] = w[1:-1, 2]
                coefs_view[11::4] = w[1:-1, 3]

            if threads > 1:
                parratt_wrapper_MT(
                    1,
                    4*nlayers + 8,
                    coefs,
                    npoints,
                    y_data,
                    x_data,
                    threads
                )
            else:
                parratt_wrapper(
                    4*nlayers + 8,
                    coefs,
                    npoints,
                    y_data,
                    x_data
                )
    finally:
        PyMem_Free(coefs)

    return y


@cython.boundscheck(False)
cpdef _contract_by_area(np.ndarray[np.float64_t, ndim=2] slabs, dA=0.5):
    newslabs = np.copy(slabs)[::-1]

    cdef:
        double [:, :] newslabs_view = newslabs
        double [:] d = newslabs_view[:, 0]
        double [:] rho = newslabs_view[:, 1]
        double [:] irho = newslabs_view[:, 2]
        double [:] sigma = newslabs[:, 3]
        double [:] vfsolv = newslabs[:, 4]

        size_t n = np.size(d, 0)
        size_t i, newi
        double dz, rhoarea, irhoarea, vfsolvarea, rholo, rhohi, irholo, irhohi
        double da = float(dA)

    with nogil:
        i = 1
        newi = 1  # skip the substrate

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
                if (rhohi - rholo) * (dz + d[i]) > da:
                    break

                if irho[i] < irholo:
                    irholo = irho[i]
                if irho[i] > irhohi:
                    irhohi = irho[i]
                if (irhohi - irholo) * (dz + d[i]) > da:
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


cpdef np.ndarray abeles_vectorised(
    np.ndarray x,
    double[:, :, :] w,
    scale=None,
    bkg=None,
    int threads=-1
):
    """
    Vectorised Abeles matrix formalism for calculating reflectivity from a
    stratified medium.

    Parameters
    ----------
    q: array_like
        the q values required for the calculation.
        Q = 4 * Pi / lambda * sin(omega).
        Units = Angstrom**-1
    layers: np.ndarray
        coefficients required for the calculation, has shape (M, 2 + N, 4).
        The calculation is vectorised over the M sets of film parameters, and
        N is the number of layers in each film.
        layers[:, 0, 1] - SLD of fronting (/1e-6 Angstrom**-2)
        layers[:, 0, 2] - iSLD of fronting (/1e-6 Angstrom**-2)
        layers[:, N, 0] - thickness of layer N
        layers[:, N, 1] - SLD of layer N (/1e-6 Angstrom**-2)
        layers[:, N, 2] - iSLD of layer N (/1e-6 Angstrom**-2)
        layers[:, N, 3] - roughness between layer N-1/N
        layers[:, -1, 1] - SLD of backing (/1e-6 Angstrom**-2)
        layers[:, -1, 2] - iSLD of backing (/1e-6 Angstrom**-2)
        layers[:, -1, 3] - roughness between backing and last layer
    scale: array-like, optional
        Multiply all reflectivities by this value.
    bkg: array-like, optional
        Linear background to be added to all reflectivities
    threads: int, optional
        How many threads you would like to use in the reflectivity calculation.
        If `threads == -1` then the calculation is automatically spread over
        `multiprocessing.cpu_count()` threads.

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """
    if w.shape[2] != 4 or w.shape[1] < 2:
        raise ValueError(
            "Layer parameters for _creflect.abeles_vectorised must be an"
            "array of shape (>=1, >2, 4)"
        )
    if x.dtype != np.float64:
        raise ValueError("Q values for _creflect.abeles_vectorised must be np.float64")

    if scale is not None:
        if not isinstance(scale, np.ndarray) or scale.shape[0] != w.shape[0]:
            raise ValueError("scale must be an array of shape (M,)")
    else:
        scale = np.ones(w.shape[0], dtype=np.float64)

    if bkg is not None:
        if not isinstance(bkg, np.ndarray) or bkg.shape[0] != w.shape[0]:
            raise ValueError("bkg must be an array of shape (M,)")
    else:
        bkg = np.zeros(w.shape[0])

    if threads == -1:
        threads = NCPU
    elif threads == 0:
        threads = 1

    cdef:
        int nlayers = w.shape[1] - 2
        int i
        int j
        int offset
        unsigned int batch = w.shape[0]
        int npoints = x.size
        np.ndarray yout = np.repeat(
            np.empty_like(x, np.float64)[np.newaxis, ...], batch, axis=0
        )
        double *x_data
        double *bkg_data
        double *scale_data
        double *y_out_data
        double *coefs
        double *coefs_arr
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x, dtype=np.float64)

    x_data = <float64_t *>np.PyArray_DATA(x)
    y_out_data = <float64_t *>np.PyArray_DATA(yout)
    scale_data = <float64_t *>np.PyArray_DATA(scale)
    bkg_data = <float64_t *> np.PyArray_DATA(bkg)

    offset = 4*nlayers + 8
    coefs_arr = <double*> PyMem_Malloc(offset * sizeof(double) * batch)
    if not coefs_arr:
        raise MemoryError()

    try:
        for i in range(batch):
            _o = offset * i
            coefs = &coefs_arr[_o]
            coefs[0] = nlayers
            coefs[1] = scale_data[i]
            coefs[2] = w[i, 0, 1]
            coefs[3] = w[i, 0, 2]
            coefs[4] = w[i, -1, 1]
            coefs[5] = w[i, -1, 2]
            coefs[6] = bkg_data[i]
            coefs[7] = w[i, -1, 3]

            if nlayers:
                for j in range(nlayers):
                    coefs[8 + 4*j] = w[i, j + 1, 0]
                    coefs[9 + 4*j] = w[i, j + 1, 1]
                    coefs[10 + 4*j] = w[i, j + 1, 2]
                    coefs[11 + 4*j] = w[i, j + 1, 3]

        abeles_wrapper_MT(
            batch,
            4*nlayers + 8,
            coefs_arr,
            npoints,
            y_out_data,
            x_data,
            threads
        )
    finally:
        PyMem_Free(coefs_arr)

    return yout


cpdef np.ndarray gepore(
    np.ndarray x,
    double[:, :] w,
    double H=0,
    double scale=1.0,
    double bkg=0.,
    int threads=-1,
):
    """
    Abeles matrix formalism for calculating polarised neutron reflectivity
    from a stratified medium.

    Parameters
    ----------
    q: array_like
        the q values required for the calculation.
        Q = 4 * Pi / lambda * sin(omega).
        Units = Angstrom**-1
    layers: np.ndarray
        coefficients required for the calculation, has shape (2 + N, 6),
        where N is the number of layers
        layers[0, 1] - SLD of fronting (/1e-6 Angstrom**-2)
        layers[0, 2] - iSLD of fronting (/1e-6 Angstrom**-2)
        layers[N, 0] - thickness of layer N
        layers[N, 1] - SLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 2] - iSLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 3] - roughness between layer N-1/N
        layers[N, 4] - magnetic SLD (/1e-6 Angstrom**-2)
        layers[N, 5] - angle of the magnetism within the layer (degrees)
        layers[-1, 1] - SLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 2] - iSLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 3] - roughness between backing and last layer
    H: float
        Applied magnetic field (T)
    scale: float
        Multiply all reflectivities by this value.
    bkg: float
        Linear background to be added to all reflectivities

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value. Has shape `(4, M)`
        where M is the number of points in `q`.
    """
    if w.shape[1] != 6 or w.shape[0] < 2:
        raise ValueError(
            "Layer parameters for _creflect.gepore must be an"
            "array of shape (>=2, 6)"
        )
    if x.dtype != np.float64:
        raise ValueError("Q values for _creflect.gepore must be np.float64")

    cdef:
        int npoints = x.size
        np.ndarray y = np.zeros((4, npoints), np.complex128)
        double *d_data
        double *sigma_data
        double *rho_data
        double *irho_data
        double *rhoM_data
        double *thetaM_data
        Cplx *Ra
        Cplx *Rb
        Cplx *Rc
        Cplx *Rd
        const double *kz_data
        int layers = w.shape[0]
        d = np.zeros(layers)
        sigma = np.zeros(layers - 1)
        rho = np.zeros(layers)
        irho = np.zeros(layers)
        rhoM = np.empty(layers)
        thetaM = np.empty(layers)
        double _rhoM

        u1 = np.empty(layers, np.complex128)
        u3 = np.empty(layers, np.complex128)
        Cplx U1
        Cplx U3
        Cplx *u1_data
        Cplx *u3_data

    d[:] = w[:, 0]
    sigma[:] = w[1:, 3]
    rho[:] = w[:, 1]
    irho[:] = w[:, 2]
    rhoM[:] = w[:, 4]
    thetaM[:] = w[:, 5]

    d_data = <float64_t *>np.PyArray_DATA(d)
    sigma_data = <float64_t *>np.PyArray_DATA(sigma)
    rho_data = <float64_t *>np.PyArray_DATA(rho)
    irho_data = <float64_t *>np.PyArray_DATA(irho)
    rhoM_data = <float64_t *>np.PyArray_DATA(rhoM)
    thetaM_data = <float64_t *>np.PyArray_DATA(thetaM)
    u1_data = <Cplx *> np.PyArray_DATA(u1)
    u3_data = <Cplx *> np.PyArray_DATA(u3)

    Ra = <Cplx *> np.PyArray_DATA(y[0])
    Rb = <Cplx *> np.PyArray_DATA(y[1])
    Rc = <Cplx *> np.PyArray_DATA(y[2])
    Rd = <Cplx *> np.PyArray_DATA(y[3])

    for idx in range(layers):
        _rhoM = w[idx, 4]
        calculate_U1_U3(H, _rhoM, w[idx, 5] * np.pi / 180., -90, U1, U3)
        rhoM[idx] = _rhoM
        u1[idx] = U1
        u3[idx] = U3

    kz = x / 2.0
    kz_data = <float64_t *> np.PyArray_DATA(kz)

    magnetic_amplitude(layers, d_data, sigma_data, rho_data, irho_data, rhoM_data, u1_data, u3_data, npoints, kz_data, NULL, Ra, Rb, Rc, Rd)

    out = y * np.conj(y)
    out *= scale
    out += bkg
    return out.real
