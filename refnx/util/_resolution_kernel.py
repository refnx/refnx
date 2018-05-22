#!/usr/bin/python
"""
Resolution kernel for a conventional double disc chopper system
"""
from __future__ import division
import numpy as np
from scipy import stats, integrate, constants, optimize

import refnx.util.general as general

# h / m = 3956
K = constants.h / constants.m_n * 1.e10


"""
ANGULAR COMPONENTS
"""


class P_Theta(object):
    def __init__(self, d1, d2, L12=2859.):
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

        alpha = (d1 + d2) / 2. / L12
        beta = abs(d1 - d2) / 2. / L12

        self.alpha = alpha
        self.beta = beta

        c = (alpha - beta) / 2 / alpha
        d = (alpha + beta) / 2 / alpha
        self.rv = stats.trapz(c, d, -alpha, 2 * alpha)
        self.width = alpha

    def __call__(self, theta):
        """
        Calculates pdf(theta) for the collimation system

        Parameters
        ----------
        theta: float
            quantiles, in radians

        Returns
        -------
        pdf: float
            Probability density function at theta
        """
        return self.rv.pdf(theta)


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
    pdf = wavelength0 / 4 / np.pi / np.cos(theta) * p_theta(theta)
    pdf /= integrate.simps(pdf, x=Q)
    return pdf


"""
Wavelength components
"""


class P_Wavelength(object):
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
        tc = general.tauC(wavelength0, xsi=self.xsi, z0=self.z0 / 1000,
                          freq=self.freq)
        # time of flight
        TOF = self.L / 1000. / general.wavelength_velocity(wavelength0)

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
        tc = general.tauC(wavelength0, xsi=self.xsi, z0=self.z0 / 1000,
                          freq=self.freq)
        # time of flight
        TOF = self.L / 1000. / general.wavelength_velocity(wavelength0)

        width = tc / TOF * wavelength0
        return stats.uniform.pdf(wavelength - wavelength0,
                                 -width / 2, width)

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
        return stats.uniform.pdf(wavelength - wavelength0,
                                 -width / 2, width)

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
        TOF = self.L / 1000. / general.wavelength_velocity(wavelength0)

        width = tau_h / TOF * wavelength0
        return stats.uniform.pdf(wavelength - wavelength0,
                                 -width / 2, width)


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
    pdf /= integrate.simps(pdf, Q)

    return pdf


def resolution_kernel(p_theta, p_wavelength, theta0, wavelength0,
                      npnts=1001, spectrum=None):
    mean_q = general.q(theta0, wavelength0)
    max_q = general.q(theta0 + p_theta.width,
                      wavelength0 - p_wavelength.width(wavelength0))
    width = max_q - mean_q
    Q = np.linspace(mean_q - width, mean_q + width, npnts)

    pqt = pq_theta(p_theta, theta0, wavelength0, Q)
    pqb = pq_wavelength(p_wavelength.burst, theta0, wavelength0, Q, spectrum)
    pqc = pq_wavelength(p_wavelength.crossing, theta0, wavelength0, Q,
                        spectrum)
    pqda = pq_wavelength(p_wavelength.da, theta0, wavelength0, Q, spectrum)

    spacing = np.diff(Q)[0]

    kernel = np.convolve(pqt, pqb, 'same')
    kernel = np.convolve(kernel, pqc, 'same')
    kernel = np.convolve(kernel, pqda, 'same')
    kernel *= spacing ** 3.

    # ensure that it's normalised. Generally for lots and lots of points the
    # normalisation approaches 1 anyway, but that takes computational time.
    return Q, kernel / integrate.simps(kernel, Q)
