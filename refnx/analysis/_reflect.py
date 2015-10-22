from __future__ import division, print_function
import numpy as np


def abeles(q, layers, scale=1., bkg=0, parallel=True):
    """
    Abeles matrix formalism for calculating reflectivity from a stratified
    medium.

    Parameters
    ----------
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
    q: array_like
        the q values required for the calculation.
        Q = 4 * Pi / lambda * sin(omega).
        Units = Angstrom**-1
    scale: float
        Multiply all reflectivities by this value.
    bkg: float
        Linear background to be added to all reflectivities
    parallel: bool
        <THIS OPTION IS CURRENTLY IGNORED>

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """

    qvals = np.asfarray(q).ravel()
    nlayers = layers.shape[0] - 2
    npnts = qvals.size

    kn = np.zeros((npnts, nlayers + 2), np.complex128)

    sld = np.zeros(nlayers + 2, np.complex128)
    sld[:] += ((layers[:, 1] - layers[0, 1]) + 1j * (layers[:, 2] - layers[0, 2])) * 1.e-6

    # kn is a 2D array. Rows are Q points, columns are kn in a layer.
    kn[:] = np.sqrt(qvals[:, np.newaxis]**2. / 4. - 4. * np.pi * sld)

    # work out the fresnel reflection for each layer
    rj = (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:])
    rj *= np.exp(kn[:, :-1] * kn[:, 1:] * -2. * layers[1:, 3]**2)

    beta = np.ones((npnts, layers.shape[0] - 1), np.complex128)

    if nlayers:
        beta[:, 1:] = np.exp(kn[:, 1: -1] * 1j * np.fabs(layers[1: -1, 0]))

    mrtotal = np.zeros((npnts, 2, 2), np.complex128)
    mi = np.zeros((npnts, nlayers + 1, 2, 2), np.complex128)
    mi[:, :, 0, 0] = beta
    mi[:, :, 1, 1] = 1. / beta
    mi[:, :, 0, 1] = rj * beta
    mi[:, :, 1, 0] = rj * mi[:, :, 1, 1]

    mrtotal[:] = mi[:, 0]

    for layer in range(1, nlayers + 1):
        mrtotal = np.einsum('...ij,...jk->...ik', mrtotal, mi[:, layer])

    # now work out the reflectivity
    reflectivity = ((mrtotal[:, 1, 0] * np.conj(mrtotal[:, 1, 0])) /
                    (mrtotal[:, 0, 0] * np.conj(mrtotal[:, 0, 0])))

    reflectivity *= scale
    reflectivity += bkg
    return np.real(np.reshape(reflectivity, q.shape))


if __name__ == '__main__':
    a = np.zeros(12)
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
        abeles(b, a)

    for i in range(1000):
        loop()
