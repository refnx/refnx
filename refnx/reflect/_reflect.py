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
        # mi11 = (np.exp(k * -1j * np.fabs(layers[idx - 1, 0]))
        #         if idx - 1 else 1)
        mi11 = 1 / mi00 if idx - 1 else 1

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


"""
PNR calculation
"""
def pmatrix(kn_u, kn_d, thickness):
    # equation 7 + 14 in Blundell and Bland
    P = np.zeros((kn_u.size, 4, 4), np.complex128)

    p0 = np.exp(complex(0, 1) * kn_u * thickness)
    p1 = np.exp(complex(0, 1) * kn_d * thickness)

    P[:, 0, 0] = 1 / p0
    P[:, 1, 1] = p0
    P[:, 2, 2] = 1 / p1
    P[:, 3, 3] = p1

    return P


def dmatrix(kn_u, kn_d):
    # equation 5 + 13 in Blundell and Bland
    D = np.zeros((kn_u.size, 4, 4), np.complex128)

    D[:, 0, 0] = 1
    D[:, 0, 1] = 1
    D[:, 1, 0] = kn_u
    D[:, 1, 1] = -kn_u

    D[:, 2, 2] = 1
    D[:, 2, 3] = 1
    D[:, 3, 2] = kn_d
    D[:, 3, 3] = -kn_d

    return D


def rmatrix(theta):
    # equation 15 in Blundell and Bland
    R = np.zeros((4, 4), np.complex128)

    cos_term = np.cos(theta / 2.) * complex(1, 0)
    sin_term = np.sin(theta / 2.) * complex(1, 0)

    R[0, 0] = cos_term
    R[1, 1] = cos_term

    R[0, 2] = sin_term
    R[1, 3] = sin_term

    R[2, 0] = -sin_term
    R[3, 1] = -sin_term

    R[2, 2] = cos_term
    R[3, 3] = cos_term

    return R


def magsqr(z):
   """
   Return the magnitude squared of the real- or complex-valued input.
   """
   return np.abs(z)**2


def pnr(q, layers):
    """
    layers
    [[thick_n, sld_n, isld_n, magsld_n, theta_n]]

    """
    xx = np.asfarray(q).astype(np.complex128).ravel()

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
    kn_u = np.sqrt(0.25 * xx[:, np.newaxis]**2 - 4 * np.pi * sldu)
    kn_d = np.sqrt(0.25 * xx[:, np.newaxis]**2 - 4 * np.pi * sldd)

    MM = np.zeros((xx.size, 4, 4), np.complex128)
    MM[:] = np.identity(4, np.complex128)

    # iterate over layers
    for jj in range(len(layers) - 2):
        R = rmatrix(thetas[jj + 1])
        D = dmatrix(kn_u[:, jj + 1], kn_d[:, jj + 1])

        P = pmatrix(kn_u[:, jj + 1], kn_d[:, jj + 1], layers[jj + 1, 0])
        MM = MM @ D @ P @ np.linalg.inv(D) @ R

    R = rmatrix(thetas[0])
    D = dmatrix(kn_u[:, 0], kn_d[:, 0])

    M = np.linalg.inv(D) @ R @ MM @ dmatrix(kn_u[:, -1], kn_d[:, -1])

    # equation 16 in Blundell and Bland
    den = (M[:, 0, 0] * M[:, 2, 2] - M[:, 0, 2] * M[:, 2, 0])
    pp = magsqr((M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0]) / den) # uu

    mm = magsqr((M[:, 3, 2] * M[:, 0, 0] - M[:, 3, 0] * M[:, 0, 2]) / den) # dd

    pm = magsqr((M[:, 3, 0] * M[:, 2, 2] - M[:, 3, 2] * M[:, 2, 0]) / den) # ud

    mp = magsqr((M[:, 1, 2] * M[:, 0, 0] - M[:, 1, 0] * M[:, 0, 2]) / den) # du

    return (pp, mm, pm, mp)


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
