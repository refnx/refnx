# cython: language_level=3, boundscheck=False

from libc.stdlib cimport abort, malloc, free
import numpy as np
cimport numpy as np
cimport cython
cimport openmp
from cython.parallel import prange, parallel, threadid


np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
TINY = 1e-30


cdef extern from "<complex>" namespace "std" nogil:
    double complex sqrt(double complex)
    double complex exp(double complex)
    double complex conj(double complex)


cdef extern from "<math.h>" nogil:
    double fabs(double)


cdef extern from "refcaller.h" nogil:
    void abeles_wrapper(int numcoefs, const double *coefP, int npoints, double *yP,
                 const double *xP)


cpdef abeles(x, np.ndarray[DTYPE_t, ndim=2] w,
             double scale=1.0, double bkg=0., int threads=-1):

    # we need the abscissae in a contiguous block of memory
    cdef double[:] xtemp = np.ascontiguousarray(x, dtype=DTYPE).flatten()

    if w.shape[1] != 4 or w.shape[0] < 2:
        raise ValueError("Parameters for reflection must have shape (>2, 4)")

    if threads > 0:
        num_threads = int(threads)
    else:
        num_threads = openmp.omp_get_max_threads()

    y = _abeles(xtemp, w, scale, bkg, num_threads)

    y = np.reshape(y, x.shape)
    return y


cpdef parratt(x, np.ndarray[DTYPE_t, ndim=2] w,
             double scale=1.0, double bkg=0., int threads=-1):

    # we need the abscissae in a contiguous block of memory
    cdef double[:] xtemp = np.ascontiguousarray(x, dtype=DTYPE).flatten()

    if w.shape[1] != 4 or w.shape[0] < 2:
        raise ValueError("Parameters for reflection must have shape (>2, 4)")

    if threads > 0:
        num_threads = int(threads)
    else:
        num_threads = openmp.omp_get_max_threads()

    y = _parratt(xtemp, w, scale, bkg, num_threads)

    y = np.reshape(y, x.shape)
    return y


@cython.boundscheck(False)
@cython.cdivision(True)
cdef _abeles(double[:] x,
             np.ndarray[np.float64_t, ndim=2] w,
             double scale=1.0, double bkg=0., int num_threads=1):

    cdef:
        int _num_threads = num_threads
        int nlayers = w.shape[0] - 2
        int npoints = x.shape[0]
        int layer, i, m
        double complex M_4PI = 4 * np.pi
        double complex I = 1j
        double complex rj, k, k_next, q2, rough, mi00, mi01, mi10, mi11, thick
        double complex mrtot00, mrtot01, mrtot10, mrtot11, p0, p1, beta, arg

        np.ndarray[np.complex128_t, ndim=1] y = np.zeros(npoints,
                                                         np.complex128)
        np.ndarray[np.complex128_t, ndim=1] roughsqr = np.empty(
            nlayers + 1, np.complex128)

        np.ndarray[np.complex128_t, ndim=1] SLD = np.zeros(
            (w.shape[0]), np.complex128)

        double[:, :] wbuf = w
        double complex[:] SLDbuf = SLD
        double complex[:] roughbuf = roughsqr

    for i in range(1, nlayers + 2):
        SLDbuf[i] = M_4PI * (wbuf[i, 1] - wbuf[0, 1] +
                             1j * (fabs(wbuf[i, 2]) + TINY)) * 1.e-6
        roughbuf[i - 1] = -2. * wbuf[i, 3] * wbuf[i, 3]

    for i in prange(npoints, nogil=True, num_threads=_num_threads):
        q2 = x[i] * x[i] / 4.
        k = x[i] / 2.
        for m in range(0, nlayers + 1):
            k_next = sqrt(q2 - SLDbuf[m + 1])
            rj = (k - k_next) / (k + k_next) * exp(k * k_next * roughbuf[m])

            if not m:
                # characteristic matrix for first interface
                mrtot00 = 1.
                mrtot01 = rj
                mrtot11 = 1.
                mrtot10 = rj
            else:
                # work out the beta for the layer
                thick = wbuf[m, 0]
                beta = exp(k * thick * I)
                # this is the characteristic matrix of a layer
                mi00 = beta
                mi11 = 1. / beta
                mi10 = rj * mi00
                mi01 = rj * mi11

                # matrix multiply
                p0 = mrtot00 * mi00 + mrtot10 * mi01
                p1 = mrtot00 * mi10 + mrtot10 * mi11
                mrtot00 = p0
                mrtot10 = p1

                p0 = mrtot01 * mi00 + mrtot11 * mi01
                p1 = mrtot01 * mi10 + mrtot11 * mi11

                mrtot01 = p0
                mrtot11 = p1

            k = k_next

        y[i] = (mrtot01 / mrtot00)
        y[i] = y[i] * conj(y[i])
        y[i] *= scale
        y[i] += bkg
    return y.real


@cython.boundscheck(False)
@cython.cdivision(True)
cdef _parratt(double[:] x,
              np.ndarray[np.float64_t, ndim=2] w,
              double scale=1.0, double bkg=0., int num_threads=1):

    cdef:
        int _num_threads = num_threads
        int nlayers = w.shape[0] - 2
        int npoints = x.shape[0]
        int layer, i, idx
        double complex M_4PI = 4 * np.pi
        double complex I = 1j
        double complex rj, kn, kn_next, qq2, rough
        double complex beta, arg, RRJ, RRJ_1

        np.ndarray[np.complex128_t, ndim=1] y = np.zeros(npoints,
                                                         np.complex128)
        np.ndarray[np.complex128_t, ndim=1] roughsqr = np.empty(
            nlayers + 1, np.complex128)

        np.ndarray[np.complex128_t, ndim=1] SLD = np.zeros(
            (w.shape[0]), np.complex128)

        double[:, :] wbuf = w
        double complex[:] SLDbuf = SLD
        double complex[:] roughbuf = roughsqr

    for idx in range(1, nlayers + 2):
        SLDbuf[idx] = M_4PI * (wbuf[idx, 1] - wbuf[0, 1] +
                               I * (fabs(wbuf[idx, 2]) + TINY)) * 1.e-6
        roughbuf[idx - 1] = -2. * wbuf[idx, 3] * wbuf[idx, 3]

    for i in prange(npoints, nogil=True, num_threads=_num_threads):
        qq2 = x[i] * x[i] / 4.

        # start from subphase
        kn_next = sqrt(qq2 - SLD[nlayers + 1])

        for idx in range(nlayers, -1, -1):
            # wavevector in the layer
            kn = sqrt(qq2 - SLD[idx])

            # reflectance of the interface
            # factor of 2 is already incorporated in rough_sqr
            rj = ((kn - kn_next)/(kn + kn_next)
                  * exp(kn * kn_next * roughbuf[idx]))

            if (idx == nlayers):
                # characteristic matrix for first interface
                RRJ = rj
            else:
                beta = exp(-2.0 * I * kn_next * fabs(wbuf[idx + 1, 0]))
                RRJ = (rj + RRJ_1 * beta) / (1 + RRJ_1 * beta * rj)

            kn_next = kn
            RRJ_1 = RRJ

        y[i] = RRJ_1 * conj(RRJ_1)
        y[i] *= scale
        y[i] += bkg
    return y.real


cpdef np.ndarray abeles_vectorised(
    np.ndarray x,
    double[:, :, :] w,
    scale=None,
    bkg=None,
    int threads=-1):
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
            "Layer parameters for _creflect.vec_abeles must be an"
            "array of shape (>=1, >2, 4)"
        )
    if x.dtype != np.float64:
        raise ValueError("Q values for _creflect.vec_abeles must be np.float64")

    if scale is not None:
        if not isinstance(scale, np.ndarray) or scale.shape != w.shape[0]:
            raise ValueError("scale must be an array of shape (M,)")
    else:
        scale = np.ones(w.shape[0], dtype=np.float64)

    if bkg is not None:
        if not isinstance(bkg, np.ndarray) or bkg.shape != w.shape[0]:
            raise ValueError("bkg must be an array of shape (M,)")
    else:
        bkg = np.zeros(w.shape[0])

    cdef:
        int nlayers = w.shape[1] - 2
        int i
        int j
        int num_threads
        int offset
        int nvec = w.shape[0]
        int npoints = x.size
        np.ndarray yout = np.repeat(
            np.empty_like(x, np.float64)[np.newaxis, ...], nvec, axis=0
        )
        double *x_data
        double *bkg_data
        double *scale_data
        double *y_out_data
        double *coefs
        double *coefs_arr
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x, dtype=np.float64)

    x_data = <DTYPE_t *>np.PyArray_DATA(x)
    y_out_data = <DTYPE_t *>np.PyArray_DATA(yout)
    scale_data = <DTYPE_t *>np.PyArray_DATA(scale)
    bkg_data = <DTYPE_t *> np.PyArray_DATA(bkg)

    if threads > 0:
        num_threads = int(threads)
    else:
        num_threads = openmp.omp_get_max_threads()

    offset = 4*nlayers + 8
    coefs_arr = <double *> malloc(offset * sizeof(double) * num_threads)
    if coefs_arr is NULL:
        abort()

    for i in prange(nvec, nogil=True, num_threads=num_threads):
        coefs = &coefs_arr[offset * threadid()]
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

        abeles_wrapper(
            4*nlayers + 8,
            coefs,
            npoints,
            &y_out_data[npoints * i],
            x_data
        )
    free(coefs_arr)

    return yout
