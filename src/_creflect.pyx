from __future__ import division, absolute_import
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
           double scale=1.0, double bkg=0., int threads=0):
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

    threads = threads or NCPU

    if threads > 1:
        reflectMT(4*nlayers + 8, <const double*>coefs.data, npoints,
                  <double*>y.data, <const double*>xtemp.data, threads)
    else:
        reflect(4*nlayers + 8, <const double*>coefs.data, npoints,
                <double*>y.data, <const double*>xtemp.data)

    return y

"""
# Slower than the python version!
@cython.boundscheck(False)
@cython.cdivision(True)
def reflect(np.ndarray[DTYPE_t, ndim=1] x,
             np.ndarray[DTYPE_t, ndim=2] w,
             double scale=1.0, double bkg=0.):
    if w.shape[1] != 4 or w.shape[0] < 2:
        raise ValueError("Parameters for reflection must have shape (>2, 4)")
    assert w.dtype == np.float64 and x.dtype == np.float64
    cdef int nlayers = w.shape[0] - 2
    cdef int npoints = x.shape[0]
    cdef int layer
    cdef np.ndarray[np.complex128_t, ndim=1] y = np.zeros(npoints,
                             np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2] kn = np.zeros(
                            (npoints, w.shape[0]), np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] rj = np.zeros(
                            (npoints), np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] SLD = np.zeros(
                            (w.shape[0]), np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] mi00 = np.ones_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] mi01 = np.zeros_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] mi10 = np.zeros_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] mi11 = np.ones_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] mrtot00 = np.ones_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] mrtot01 = np.zeros_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] mrtot10 = np.zeros_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] mrtot11 = np.ones_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] p0 = np.zeros_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] p1 = np.zeros_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] k = np.zeros_like(y, np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1] k_next = np.zeros_like(y, np.complex128)

    SLD[:] += 4 * np.pi * (w[:, 1] - w[0, 1] + 1j * (w[:, 2] - w[0, 2])) * 1.e-6
    kn[:] = np.sqrt(x.flatten()[:, np.newaxis]**2 / 4. - SLD)

    k = kn[:, 0]
    for idx in range(1, nlayers + 2):
        k_next = kn[:, idx]
        rj = (k - k_next) / (k + k_next)
        rj *= np.exp(k * k_next * -2. * w[idx, 3] ** 2)

        # work out characteristic matrix of layer
        if idx - 1:
            mi00 = np.exp(k * 1j * np.fabs(w[idx - 1, 0]))
            mi11 = np.exp(k * -1j * np.fabs(w[idx - 1, 0]))

        mi10 = rj * mi00
        mi01 = rj * mi11

        # matrix multiply mrtot and mi
        p0 = mrtot00 * mi00 + mrtot10 * mi01
        p1 = mrtot00 * mi10 + mrtot10 * mi11
        mrtot00 = p0
        mrtot10 = p1

        p0 = mrtot01 * mi00 + mrtot11 * mi01
        p1 = mrtot01 * mi10 + mrtot11 * mi11

        mrtot01 = p0
        mrtot11 = p1

        k = k_next

    y = (mrtot01 * np.conj(mrtot01)) / (mrtot00 * np.conj(mrtot00)))
    y *= scale
    y += bkg
    return y.real
"""