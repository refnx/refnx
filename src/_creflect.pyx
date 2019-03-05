# cython: language_level=3, boundscheck=False
from multiprocessing import cpu_count
import numpy as np

cimport numpy as np
cimport cython
from cython.view cimport array as cvarray

cdef extern from "refcalc.h":
    void reflect(int numcoefs, const double *coefP, int npoints, double *yP,
                 const double *xP)
    void reflectMT(int numcoefs, const double *coefP, int npoints, double *yP,
                   const double *xP, int threads)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


# figure out CPU count
NCPU = cpu_count()


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray abeles(np.ndarray x,
           np.ndarray[DTYPE_t, ndim=2] w,
           double scale=1.0, double bkg=0., int threads=-1):
    if w.shape[1] != 4 or w.shape[0] < 2:
        raise ValueError("Parameters for _creflect must have shape (>2, 4)")
    if (w.dtype != np.float64 or x.dtype != np.float64):
        raise ValueError("Parameters for _creflect must be np.float64")

    cdef int nlayers = w.shape[0] - 2
    cdef int npoints = x.size
    cdef np.ndarray[DTYPE_t, ndim=1] coefs = np.empty(4 * nlayers + 8,
                                                      DTYPE)
    cdef np.ndarray y = np.empty_like(x, DTYPE)

    # we need the abscissae in a contiguous block of memory
    cdef np.ndarray xtemp = np.ascontiguousarray(x, dtype=DTYPE)

    coefs[0] = nlayers
    coefs[1] = scale
    coefs[2:4] = w[0, 1: 3]
    coefs[4: 6] = w[-1, 1: 3]
    coefs[6] = bkg
    coefs[7] = w[-1, 3]
    if nlayers:
        coefs[8:] = w.flatten()[4: -4]

    if threads == -1:
        threads = NCPU
    elif threads == 0:
        threads = 1

    if threads > 1:
        reflectMT(4*nlayers + 8, <const double*>coefs.data, npoints,
                  <double*>y.data, <const double*>xtemp.data, threads)
    else:
        reflect(4*nlayers + 8, <const double*>coefs.data, npoints,
                <double*>y.data, <const double*>xtemp.data)

    return y

"""
cdef extern from "<complex.h>":
    double complex sqrt(double complex)
    double complex exp(double complex)
    double complex conj(double complex)
    double abs(double)


import numpy as np
cimport numpy as cnp
cimport cython
from cython.view cimport array as cvarray

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
TINY = 1e-30

@cython.boundscheck(False)
@cython.cdivision(True)
def reflect(double[:] x not None,
            cnp.ndarray[cnp.float64_t, ndim=2] w,
            double scale=1.0, double bkg=0.):
    if w.shape[1] != 4 or w.shape[0] < 2:
        raise ValueError("Parameters for reflection must have shape (>2, 4)")
    cdef int nlayers = w.shape[0] - 2
    cdef int npoints = x.shape[0]
    cdef int layer
    cdef int i
    cdef int m
    cdef double complex M_4PI = 4 * np.pi
    cdef double complex I = 1j
    cdef double complex rj, k, k_next, q2, rough, mi00, mi01, mi10, mi11, thick
    cdef double complex mrtot00, mrtot01, mrtot10, mrtot11, p0, p1, beta, arg

    cdef cnp.ndarray[cnp.complex128_t, ndim=1] y = np.zeros(npoints,
                             np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] roughsqr = np.empty(nlayers + 1,
                             np.complex128)

    cdef cnp.ndarray[cnp.complex128_t, ndim=1] SLD = np.zeros(
                            (w.shape[0]), np.complex128)

    cdef double[:, :] wbuf = w
    cdef double complex[:] SLDbuf = SLD
    cdef double complex[:] roughbuf = roughsqr

    for i in range(1, nlayers + 2):
        SLDbuf[i] = M_4PI * (wbuf[i, 1] - wbuf[0, 1] +
                             1j * (abs(wbuf[i, 2]) + TINY)) * 1.e-6
        roughbuf[i - 1] = -2. * wbuf[i, 3] * wbuf[i, 3]

    for i in range(npoints):
        q2 = x[i] * x[i] / 4.
        k = sqrt(q2)
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
"""