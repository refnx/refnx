from __future__ import division, print_function
import numpy as np


def abeles(q, w, scale=1., bkg=0):
    """

       Abeles matrix formalism for calculating reflectivity from a stratified
       medium.
       Parameters
       ----------

       w - np.ndarray
           coefficients required for the calculation, has shape (2 + N, 4),
           where N is the number of layers
           w[0, 1] - SLD of fronting (/1e-6 Angstrom**-2)
           w[0, 2] - iSLD of fronting (/1e-6 Angstrom**-2)
           w[N, 0] - thickness of layer N
           w[N, 1] - SLD of layer N (/1e-6 Angstrom**-2)
           w[N, 2] - iSLD of layer N (/1e-6 Angstrom**-2)
           w[N, 3] - roughness between layer N-1/N
           w[-1, 1] - SLD of backing (/1e-6 Angstrom**-2)
           w[-1, 2] - iSLD of backing (/1e-6 Angstrom**-2)
           w[-1, 3] - roughness between backing and last layer

        q - np.ndarray
            the qvalues required for the calculation.
            Q=4*Pi/lambda * sin(omega).
            Units = Angstrom**-1
    """

    qvals = q.flatten()
    nlayers = w.shape[0] - 2
    npnts = qvals.size

    kn = np.zeros((npnts, nlayers + 2), np.complex128)

    SLD = np.zeros(nlayers + 2, np.complex128)
    SLD[:] += ((w[:, 1] - w[0, 1]) + 1j * (w[:, 2] - w[0, 2])) * 1.e-6

    # kn is a 2D array. Rows are Q points, columns are kn in a layer.
    kn[:] = np.sqrt(qvals[:, np.newaxis]**2. / 4. - 4. * np.pi * SLD)

    # work out the fresnel reflection for each layer
    rj = (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:])
    rj *= np.exp(kn[:, :-1] * kn[:, 1:] * -2. * w[1:, 3]**2)

    beta = np.ones((npnts, w.shape[0] - 1), np.complex128)

    if nlayers:
        beta[:, 1:] = np.exp(kn[:, 1: -1] * 1j * np.fabs(w[1: -1, 0]))

    MRtotal = np.zeros((npnts, 2, 2), np.complex128)
    MI = np.zeros((npnts, nlayers + 1, 2, 2), np.complex128)
    MI[:, :, 0, 0] = beta
    MI[:, :, 1, 1] = 1. / beta
    MI[:, :, 0, 1] = rj * beta
    MI[:, :, 1, 0] = rj * MI[:, :, 1, 1]

    MRtotal[:] = MI[:, 0]

    for layer in range(1, nlayers + 1):
        MRtotal = np.einsum('...ij,...jk->...ik', MRtotal, MI[:, layer])

    # now work out the reflectivity
    reflectivity = (MRtotal[:, 1, 0] * np.conj(MRtotal[:, 1, 0])) / \
        (MRtotal[:, 0, 0] * np.conj(MRtotal[:, 0, 0]))

    reflectivity *= scale
    reflectivity += bkg
    return np.reshape(reflectivity, q.shape)


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

    b = np.arange(1000.)
    b /= 2000.
    b += 0.001

    def loop():
        abeles(len(b), b, a)

    t = timeit.Timer(stmt=loop)
    print(t.timeit(number=1000))
