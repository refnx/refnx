from __future__ import division, print_function
import numpy as np


"""
import numpy as np
q = np.linspace(0.01, 0.5, 1000)
w = np.array([[0, 2.07, 0, 0],
              [100, 3.47, 0, 3],
              [500, -0.5, 0.00001, 3],
              [0, 6.36, 0, 3]])
"""


def abeles(q, layers, scale=1., bkg=0, threads=0):
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
    threads: int, optional
        <THIS OPTION IS CURRENTLY IGNORED>

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """
    qvals = np.asfarray(q)
    flatq = qvals.ravel()

    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    kn = np.zeros((npnts, nlayers + 2), np.complex128)

    sld = np.zeros(nlayers + 2, np.complex128)
    sld[:] += ((layers[:, 1] - layers[0, 1]) +
               1j * (layers[:, 2] - layers[0, 2])) * 1.e-6

    # kn is a 2D array. Rows are Q points, columns are kn in a layer.
    # calculate wavevector in each layer, for each Q point.
    kn[:] = np.sqrt(flatq[:, np.newaxis] ** 2. / 4. - 4. * np.pi * sld)

    # initialise matrix total
    mrtot00 = 1
    mrtot11 = 1
    mrtot10 = 0
    mrtot01 = 0
    k = kn[:, 0]

    for idx in range(1, nlayers + 2):
        k_next = kn[:, idx]

        # reflectance of an interface
        rj = (k - k_next) / (k + k_next)
        rj *= np.exp(k * k_next * -2. * layers[idx, 3] ** 2)

        # work out characteristic matrix of layer
        mi00 = np.exp(k * 1j * np.fabs(layers[idx - 1, 0])) if idx - 1 else 1
        mi11 = np.exp(k * -1j * np.fabs(layers[idx - 1, 0])) if idx - 1 else 1

        mi10 = rj * mi00
        mi01 = rj * mi11

        # matrix multiply mrtot by characteristic matrix
        p0 = mrtot00 * mi00 + mrtot10 * mi01
        p1 = mrtot00 * mi10 + mrtot10 * mi11
        mrtot00 = p0
        mrtot10 = p1

        p0 = mrtot01 * mi00 + mrtot11 * mi01
        p1 = mrtot01 * mi10 + mrtot11 * mi11

        mrtot01 = p0
        mrtot11 = p1

        k = k_next

    reflectivity = (mrtot01 * np.conj(mrtot01)) / (mrtot00 * np.conj(mrtot00))
    reflectivity *= scale
    reflectivity += bkg
    return np.real(np.reshape(reflectivity, qvals.shape))


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
