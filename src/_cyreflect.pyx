# cython: language_level=3, boundscheck=False

import numpy as np
cimport numpy as cnp
cimport cython
cimport openmp
from cython.parallel import prange


DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
TINY = 1e-30


cdef extern from "<complex>" namespace "std" nogil:
    double complex sqrt(double complex)
    double complex exp(double complex)
    double complex conj(double complex)


cdef extern from "<math.h>" nogil:
    double fabs(double)


cpdef abeles(x, cnp.ndarray[DTYPE_t, ndim=2] w,
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


@cython.boundscheck(False)
@cython.cdivision(True)
cdef _abeles(double[:] x,
             cnp.ndarray[cnp.float64_t, ndim=2] w,
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

        cnp.ndarray[cnp.complex128_t, ndim=1] y = np.zeros(npoints,
                                                           np.complex128)
        cnp.ndarray[cnp.complex128_t, ndim=1] roughsqr = np.empty(
            nlayers + 1, np.complex128)

        cnp.ndarray[cnp.complex128_t, ndim=1] SLD = np.zeros(
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
