#!/usr/bin/python
"""
Full resolution kernel for a conventional double disc chopper system
This code compares well to the kernel calculated using MCMC in:
github.com/refnx/refnx-models/blob/master/platypus-simulate/tof_simulator.py
"""
import numpy as np
from scipy import stats, integrate, constants, optimize
from scipy._lib._util import check_random_state

import refnx.util.general as general

# h / m = 3956
K = constants.h / constants.m_n * 1.0e10


"""
ANGULAR COMPONENTS
"""


class AngularDivergence:
    rv = None
    width = None
    alpha = None
    beta = None
    dtheta = None
    L12 = None

    def pdf(self, theta):
        raise NotImplementedError

    def rvs(self, size=1, random_state=None):
        # grab a random number generator
        rng = check_random_state(random_state)
        return self.rv.rvs(size=size, random_state=rng)


class P_Theta(AngularDivergence):
    """
    Calculates the angular resolution for a two slit collimation
    system.

    Parameters
    ----------
    d1 : float
        slit 1 opening
    d2 : float
        slit 2 opening
    L12 : float
        distance between slits
    """

    def __init__(self, d1, d2, L12=2859.0):
        div, alpha, beta = general.div(d1, d2, L12=L12)
        # alpha = (d1 + d2) / 2.0 / L12
        # beta = abs(d1 - d2) / 2.0 / L12

        self.alpha = alpha
        self.beta = beta
        self.dtheta = div
        self.L12 = L12
        self.d1 = d1
        self.d2 = d2

        c = (alpha - beta) / 2 / alpha
        d = (alpha + beta) / 2 / alpha
        self.rv = stats.trapezoid(c, d, -alpha, 2 * alpha)
        self.width = alpha

    def pdf(self, theta):
        """
        Calculates pdf(theta) for the collimation system

        Parameters
        ----------
        theta: float
            quantiles, in *degrees*

        Returns
        -------
        pdf: float
            Probability density function at theta
        """
        return self.rv.pdf(theta)


class P_Theta_Optimized(P_Theta):
    """
    Optimise slit settings for a given angular resolution, and a given
    footprint.

    footprint: float
        maximum footprint onto sample (mm)
    resolution: float
        fractional dtheta/theta resolution (FWHM)
    angle: float, optional
        angle of incidence in degrees
    """

    def __init__(self, footprint, dtheta, theta, L12=2859.0, L2S=120.0):
        self.footprint = footprint
        self.dtheta = theta
        self.theta = theta
        d1, d2 = general.slit_optimiser(
            footprint, dtheta, angle=theta, L12=L12, L2S=L2S, verbose=False
        )
        super(P_Theta_Optimized, self).__init__(d1, d2, L12=L12)


def pq_theta(p_theta, theta0, wavelength0, Q):
    """
    Angular component of Q resolution for a two slit collimation system

    Parameters
    ----------
    p_theta: P_Theta
        PDF of angular resolution
    theta0: float
        nominal angle of incidence, degrees
    wavelength0: float
        nominal wavelength
    Q: float
        Q value to evaluate PDF at

    Returns
    -------
    pdf: float
        Probability density function of resolution function at Q.
    """
    theta = np.radians(general.angle(Q, wavelength0) - theta0)
    pdf = (
        wavelength0
        / 4
        / np.pi
        / np.cos(theta)
        * p_theta.pdf(np.degrees(theta))
    )
    pdf /= integrate.simpson(pdf, x=Q)
    return pdf


"""
Wavelength components
"""


class P_Wavelength:
    def __init__(self, z0, L, freq, H, xsi=0, R=350, da=0.02):
        """
        Calculates the wavelength resolution for a double disc chopper system.

        Parameters
        ----------
        z0: float
            distance between choppers, mm
        L: float
            time of flight distance, mm
        freq: float
            frame frequency for double chopper disc system, Hz
        H: float
            height of beam, mm
        xsi: float
            phase opening of disc, degrees
        R: float
            radius of chopper disc, mm
        da: float
            Fractional value for wavelength rebinning
        """
        self.z0 = z0
        self.L = L
        self.freq = freq
        self.xsi = xsi
        self.H = H
        self.R = R
        self._da = da

    def width(self, wavelength0):
        # burst width
        tc = general.tauC(
            wavelength0, xsi=self.xsi, z0=self.z0 / 1000, freq=self.freq
        )
        # time of flight
        TOF = self.L / 1000.0 / general.wavelength_velocity(wavelength0)

        width = tc / TOF * wavelength0

        # da width
        width += self._da * wavelength0

        # crossing width
        tau_h = self.H / self.R / (2 * np.pi * self.freq)
        width += tau_h / TOF * wavelength0

        return width

    def burst(self, wavelength0, wavelength):
        """
        Calculates pdf(wavelength) for burst time component

        Parameters
        ----------
        wavelength0: float
            Nominal wavelength, Angstrom
        wavelength: float
            Wavelength, Angstrom

        Returns
        -------
        pdf: float
            Probability density function at `wavelength`
        """
        # burst time
        tc = general.tauC(
            wavelength0, xsi=self.xsi, z0=self.z0 / 1000, freq=self.freq
        )
        # time of flight
        TOF = self.L / 1000.0 / general.wavelength_velocity(wavelength0)

        width = tc / TOF * wavelength0
        return stats.uniform.pdf(wavelength - wavelength0, -width / 2, width)

    def da(self, wavelength0, wavelength, da=None):
        """
        Calculates pdf(wavelength) due to rebinning

        Parameters
        ----------
        wavelength0: float
            Nominal wavelength, Angstrom
        wavelength: float
            Wavelength, Angstrom
        da: float or None
            fractional bin width, Angstrom.
            If specified then it overrides the value of da provided to
            construct the class. Otherwise the value of da used to construct
            the class is used.

        Returns
        -------
        pdf: float
            Probability density function at `wavelength`
        """
        if da is not None:
            width = da * wavelength0
        else:
            width = self._da * wavelength0
        return stats.uniform.pdf(wavelength - wavelength0, -width / 2, width)

    def crossing(self, wavelength0, wavelength):
        """
        Calculates pdf(wavelength) due to crossing time

        Parameters
        ----------
        wavelength0: float
            nominal wavelength, Angstrom
        wavelength: float
            Wavelength, Angstrom

        Returns
        -------
        pdf: float
            Probability density function at `wavelength`
        """
        tau_h = self.H / self.R / (2 * np.pi * self.freq)
        TOF = self.L / 1000.0 / general.wavelength_velocity(wavelength0)

        width = tau_h / TOF * wavelength0
        return stats.uniform.pdf(wavelength - wavelength0, -width / 2, width)


def pq_wavelength(p_wavelength, theta0, wavelength0, Q, spectrum=None):
    """
    Wavelength component of Q resolution.

    Parameters
    ----------
    p_wavelength: P_Wavelength
        PDF of wavelength resolution
    theta0: float
        nominal angle of incidence, degrees
    wavelength0: float
        nominal wavelength
    Q: float
        Q value to evaluate PDF at
    spectrum: callable
        Function, spectrum(wavelength) that specifies the intensity of
        the neutron spectrum at a given wavelength

    Returns
    -------
    pdf: float
        Probability density function of resolution function at Q.
    """
    f = spectrum or (lambda x: x)

    wavelength = general.wavelength(Q, theta0)
    pdf = 4 * np.pi * p_wavelength(wavelength0, wavelength)
    pdf *= f(wavelength) / Q / Q

    # spectrum function may not be normalised.
    pdf /= integrate.simpson(pdf, x=Q)

    return pdf


def resolution_kernel(
    p_theta, p_wavelength, theta0, wavelength0, npnts=1001, spectrum=None
):
    """
    Creates a full resolution kernel based on angular and wavelength components

    Parameters
    ----------
    p_theta: P_Theta
        Angular component
    p_wavelength: P_Wavelength
        Wavelength components
    theta0: array-like
        Nominal angle of incidence, degrees
    wavelength0: array-like
        Nominal wavelength, Angstrom
    npnts: float
        number of points in the resolution kernel
    spectrum: None or callable
        Function, spectrum(wavelength) that specifies the intensity of
        the neutron spectrum at a given wavelength

    Returns
    -------
    kernel: np.ndarray
        Full resolution kernel. Has shape `(N, 2, npnts)` where `N` is the
        number of points in theta0/wavelength0.
        kernel[:, 0, :] and kernel[:, 1, :] correspond to `Q` and `PDF(Q)` for
        each of the data points in the first dimension.
    """
    theta0_arr = np.asarray(theta0).astype(float, copy=False).ravel()
    wavelength0_arr = np.asarray(wavelength0).astype(float, copy=False).ravel()
    qpnts = max(theta0_arr.size, wavelength0_arr.size)
    arr = [
        np.array(a) for a in np.broadcast_arrays(theta0_arr, wavelength0_arr)
    ]
    theta0_arr = arr[0]
    wavelength0_arr = arr[1]

    mean_q = general.q(theta0_arr, wavelength0_arr)
    max_q = general.q(
        theta0_arr + p_theta.width,
        wavelength0_arr - p_wavelength.width(wavelength0_arr),
    )
    width = max_q - mean_q

    kernel = np.zeros((qpnts, 2, npnts))

    for i in range(qpnts):
        Q = np.linspace(mean_q[i] - width[i], mean_q[i] + width[i], npnts)
        kernel[i, 0, :] = Q

        # angular component
        pqt = pq_theta(p_theta, theta0_arr[i], wavelength0_arr[i], Q)

        # burst time component
        pqb = pq_wavelength(
            p_wavelength.burst, theta0_arr[i], wavelength0_arr[i], Q, spectrum
        )

        # crossing time component
        pqc = pq_wavelength(
            p_wavelength.crossing,
            theta0_arr[i],
            wavelength0_arr[i],
            Q,
            spectrum,
        )

        # rebinning component
        pqda = pq_wavelength(
            p_wavelength.da, theta0_arr[i], wavelength0_arr[i], Q, spectrum
        )

        spacing = np.diff(Q)[0]

        p = np.convolve(pqt, pqb, "same")
        p = np.convolve(p, pqc, "same")
        p = np.convolve(p, pqda, "same")
        p *= spacing**3.0

        kernel[i, 1, :] = p / integrate.simpson(p, x=Q)

    return kernel
