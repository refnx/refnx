from __future__ import division
import numpy as np

cimport numpy as np
cimport cython

cdef extern from "refcalc.h":
    void reflect(int numcoefs, const double *coefP, int npoints, double *yP,
                const double *xP)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def abeles(np.ndarray[DTYPE_t, ndim=1] x,
           np.ndarray[DTYPE_t, ndim=2] w,
           double scale=1.0, bkg=0.):
    if w.shape[1] != 4 or w.shape[0] < 2:
        raise ValueError("Parameters for reflection must have shape (>2, 4)")
    assert w.dtype == np.float64 and x.dtype == np.float64

    cdef int nlayers = w.shape[0] - 2
    cdef int npoints = x.shape[0]
    cdef int layer
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=1] coefs = np.empty(4 * nlayers + 8,
                                                           np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(npoints,
                                                          np.float64)

    coefs[0] = nlayers
    coefs[1] = scale
    coefs[2] = w[0, 1]
    coefs[3] = w[0, 2]
    coefs[4] = w[-1, 1]
    coefs[5] = w[-1, 2]
    coefs[6] = bkg
    coefs[7] = w[-1, 3]
    for i in range(nlayers):
        coefs[4 * i + 8] = w[i + 1, 0]
        coefs[4 * i + 9] = w[i + 1, 1]
        coefs[4 * i + 10] = w[i + 1, 2]
        coefs[4 * i + 11] = w[i + 1, 3]

    reflect(4 * nlayers + 8, <const double*>coefs.data, npoints,
           <double*>y.data, <const double*>x.data)
    return y

"""
@cython.boundscheck(False)
@cython.cdivision(False)
def reflect(np.ndarray[DTYPE_t, ndim=1] x,
            np.ndarray[DTYPE_t, ndim=2] w,
            double scale=1.0, bkg=0.):

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

    cdef np.ndarray[np.complex128_t, ndim=2] rj = np.zeros(
                            (npoints, w.shape[0] - 1), np.complex128)

    cdef np.ndarray[np.complex128_t, ndim=2] beta = np.ones(
                            (npoints, w.shape[0] - 1), np.complex128)

    cdef np.ndarray[np.complex128_t, ndim=1] SLD = np.zeros(
                            (w.shape[0]), np.complex128)

    cdef np.ndarray[np.complex128_t, ndim=3] MRtotal = np.zeros(
                            (npoints, 2, 2), np.complex128)

    cdef np.ndarray[np.complex128_t, ndim=4] MI = np.zeros(
                            (npoints, nlayers + 1, 2, 2), np.complex128)

    SLD[:] += 4 * np.pi * (w[:, 1] - w[0, 1] + 1j * (w[:, 2] - w[0, 2])) * 1.e-6

    kn[:] = np.sqrt(x.flatten()[:, np.newaxis]**2 / 4. - SLD)

    rj += (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:])
    rj *= np.exp(kn[:, :-1] * kn[:, 1:] * -2. * w[1:, 3] * w[1:, 3])

    if nlayers:
        #the last term in this statement is the thickness.
        beta[:, 1:] = np.exp(kn[:, 1::-2]
                             * 1j * np.fabs(w[1::-2, 0])[np.newaxis,])

    MI[:, :, 0, 0] = beta
    MI[:, :, 1, 1] = 1. / beta
    MI[:, :, 0, 1] = rj * beta
    MI[:, :, 1, 0] = rj * MI[:, :, 1, 1]

    #start doing the matrix multiplication.  Initialise with the first interface
    MRtotal[:] = MI[:, 0]

    for layer in range(1, nlayers + 1):
        MRtotal = np.einsum('...ij,...jk->...ik', MRtotal, MI[:, layer])

    y = ((MRtotal[:, 1, 0] * np.conj(MRtotal[:, 1, 0]))
         / (MRtotal[:, 0, 0] * np.conj(MRtotal[:, 0, 0])))

    return scale * y.real + bkg
"""