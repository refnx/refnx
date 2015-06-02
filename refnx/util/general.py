#!/usr/bin/python
from __future__ import division

import numpy as np
from scipy import stats, integrate, constants, optimize

#h / m = 3956
K = constants.h / constants.m_n * 1.e10

def div(d1, d2, L12 = 2859):
    """
    Calculate the angular resolution for a set of collimation conditions

    Parameters
    ----------
    d1: float
        slit 1 opening
    d2: float
        slit 2 opening
    L12: float
        distance between slits

    Returns
    -------
        (dtheta, alpha, beta)

    dtheta is the FWHM of the Gaussian approximation to the trapezoidal
    resolution function
    alpha is the angular divergence of the penumbra
    beta is the angular divergence of the umbra

    When calculating dtheta / theta values for resolution, then dtheta is the
    value you need to use.
    See equations 11-14 in:

    [1] de Haan, V.-O.; de Blois, J.; van der Ende, P.; Fredrikze, H.; van der
    Graaf, A.; Schipper, M.; van Well, A. A. & J., v. d. Z. ROG, the neutron
    reflectometer at IRI Delft Nuclear Instruments and Methods in Physics
    Research A, 1995, 362, 434-453
    """
    divergence = 0.68 * 0.68 * (d1 * d1 + d2 * d2) / L12 / L12
    alpha = (d1 + d2) / 2. / L12
    beta = abs(d1 - d2) / 2. / L12
    return np.degrees(np.sqrt(divergence)), np.degrees(alpha), np.degrees(beta)


def q(angle, wavelength):
    """
    Calculate Q given angle of incidence and wavelength
    angle - angle of incidence (degrees)
    wavelength -  wavelength of radiation (Angstrom)
    """
    return 4 * np.pi * np.sin(np.radians(angle)) / wavelength


def q2(omega, twotheta, phi, wavelength):
    """
    convert angles and wavelength (lambda) to Q vector.

    Parameters
    ----------
    omega: float
        angle of incidence of beam (in xy plane)
    twotheta: float
        angle between direct beam and projected reflected beam onto yz plane
    phi: float
        angle between reflected beam and yz plane.
    wavelength: float

    Returns
    -------
        (Qx, Qy, Qz)
        Momentum transfer in A**-1.

    Notes
    -----
    coordinate system:
    y - along beam direction (in small angle approximation)
    x - transverse to beam direction, in plane of sample
    z - normal to sample plane.
    the xy plane is equivalent to the sample plane.

    TODO: check
    """
    omega = np.radians(omega)
    twotheta = np.radians(twotheta)
    phi = np.radians(phi)

    qx = 2 * np.pi / wavelength * np.cos(twotheta - omega) * np.sin(phi)
    qy = (2 * np.pi / wavelength * (np.cos(twotheta - omega) * np.cos(phi)
          - np.cos(omega)))
    qz = 2 * np.pi / wavelength * (np.sin(twotheta - omega) + np.sin(omega))

    return (qx, qy, qz)


def wavelength(q, angle):
    '''
    calculate wavelength given Q vector and angle
    q - wavevector (A^-1)
    angle - angle of incidence (degrees)
    '''
    return  4. * np.pi * np.sin(angle * np.pi / 180.)/q


def angle(q, wavelength):
    '''
    calculate angle given Q and wavelength
    q - wavevector (A^-1)
    wavelength -  wavelength of radiation (Angstrom)
    '''
    return  np.asin(q / 4. / np.pi * wavelength) * 180 / np.pi


def qcrit(SLD1, SLD2):
    '''
    calculate critical Q vector given SLD of super and subphases
    SLD1 - SLD of superphase (10^-6 A^-2)
    SLD2 - SLD of subphase (10^-6 A^-2)
    '''
    return np.sqrt(16. * np.pi * (SLD2 - SLD1) * 1.e-6)


def tauC(wavelength, xsi=0, z0=0.358, freq=24):
    """
    Calculates the burst time of a double chopper pair

    Parameters
    ----------
    wavelength: float
        wavelength in Angstroms
    z0: float
        distance between chopper pair (m)
    freq: float
        rotation frequency of choppers (Hz)
    xsi: float
        phase opening of chopper pair (degrees)

    References
    ----------
    [1] A. A. van Well and H. Fredrikze, On the resolution and intensity of a
    time-of-flight neutron reflectometer, Physica B 357 (2005) 204-207
    [2] A. Nelson and C. Dewhurst, Towards a detailed resolution smearing
    kernel for time-of-flight neutron reflectometers, J. Appl. Cryst. (2013)
    46, 1338-1343
    """
    tauC = z0 / wavelength_velocity(wavelength)
    tauC += np.radians(xsi) / (2 * np.pi * freq)

    return tauC


def wavelength_velocity(wavelength):
    """
    Converts wavelength to neutron velocity

    Parameters
    ----------
    wavelength: float
        wavelength of neutron in Angstrom.

    Returns
    -------
    velocity: float
        velocity of neutron in ms**-1
    """
    return K / wavelength


def double_chopper_frequency(min_wavelength, max_wavelength, L, N=1):
   """
    Calculates the maximum frequency available for a given wavelength band
    without getting frame overlap in a chopper spectrometer.

    Parameters
    ----------
    min_wavelength: float
        minimum wavelength to be used
    max_wavelength: float
        maximum wavelength to be used
    L: float
        Flight length of instrument (m)
    N: float, optional
        number of windows in chopper pair
    """
   return K / ((max_wavelength - min_wavelength) * L * N)


def resolution_double_chopper(wavelength, z0=0.358, R=0.35, freq=24,
                              H=0.005, xsi=0, L=7.5, tau_da=0):
    """
    Calculates the fractional resolution of a double chopper pair, dl/l.

    Parameters
    ----------
    wavelength: float
        wavelength in Angstroms
    z0: float
        distance between chopper pair (m)
    R: float
        radius of chopper discs (m)
    freq: float
        rotation frequency of choppers (Hz)
    N: float
        number of windows in chopper pair
    H: float
        height of beam (m)
    xsi: float
        phase opening of chopper pair (degrees)
    L: float
        Flight length of instrument (m)
    tau_da : float
        Width of timebin (s)
    """
    TOF = L / wavelength_velocity(wavelength)
    tc = tauC(wavelength, xsi=xsi, z0=z0, freq=freq)
    tauH = H / R / (2 * np.pi * freq)
    return 0.68 * np.sqrt((tc / TOF)**2 + (tauH / TOF)**2 + (tau_da / TOF)**2)


def resolution_single_chopper(wavelength, R=0.35, freq=24, H=0.005, phi=60,
                              L=7.5):
    """
    Calculates the fractional resolution of a single chopper, dl/l.

    Parameters
    ----------
    wavelength: float
        wavelength in Angstroms
    R: float
        radius of chopper discs (m)
    freq: float
        rotation frequency of choppers (Hz)
    N: float
        number of windows in chopper
    H: float
        height of beam (m)
    phi: float
        angular opening of chopper window (degrees)
    L: float
        Flight length of instrument (m)
    """
    TOF = L / wavelength_velocity(wavelength)
    tauH = H / R / (2. * np.pi * freq)
    tauC = np.radians(phi) / (2. * np.pi * freq)
    return 0.68 * np.sqrt((tauC / TOF)**2 + (tauH / TOF)**2)


def transmission_double_chopper(wavelength, z0=0.358, R=0.35, freq=24, N=1,
                                H=0.005, xsi=0):
    """
    Calculates the transmission of a double chopper pair

    Parameters
    ----------
    wavelength: float
        wavelength in Angstroms
    z0: float
        distance between chopper pair (m)
    R: float
        radius of chopper discs (m)
    freq: float
        rotation frequency of choppers (Hz)
    N: float
        number of windows in chopper pair
    H: float
        height of beam (m)
    xsi: float
        phase opening of chopper pair (degrees)

    References
    ----------
    [1] A. A. van Well and H. Fredrikze, On the resolution and intensity of a
    time-of-flight neutron reflectometer, Physica B 357 (2005) 204-207
    [2] A. Nelson and C. Dewhurst, Towards a detailed resolution smearing
    kernel for time-of-flight neutron reflectometers, J. Appl. Cryst. (2013)
    46, 1338-1343
    """

    transmission = tauC(wavelength, xsi=xsi, z0=z0, freq=freq) * freq
    transmission += H * constants.h / (2 * np.pi * R)
    return transmission * N


def transmission_single_chopper(R=0.35, phi=60, N=1, H=0.005):
    """
    Calculates the transmission of a single chopper

    Parameters
    ----------
    R: float
        radius of chopper discs (m)
    phi: float
        angular opening of chopper disc (degrees)
    N: float
        number of windows in chopper pair
    H: float
        height of beam (m)

    References
    ----------
    [1] A. A. van Well and H. Fredrikze, On the resolution and intensity of a
    time-of-flight neutron reflectometer, Physica B 357 (2005) 204-207
    [2] A. Nelson and C. Dewhurst, Towards a detailed resolution smearing
    kernel for time-of-flight neutron reflectometers, J. Appl. Cryst. (2013)
    46, 1338-1343
    """
    return N * (np.radians(phi) * R + H) / (2 * np.pi * R)


def xraylam(energy):
    '''
    convert energy (keV) to wavelength (angstrom)
    '''
    return 12.398/ energy


def xrayenergy(wavelength):
    '''
    convert energy (keV) to wavelength (angstrom)
    '''
    return 12.398/ wavelength


def beamfrac(FWHM, length, angle):
    '''
    return the beam fraction intercepted by a sample of length length
    at sample tilt angle.
    The beam is assumed to be gaussian, with a FWHM of FWHM.
    '''
    height_of_sample = length * np.sin(np.radians(angle))
    beam_sd = FWHM / 2 / np.sqrt(2 * np.log(2))
    probability = 2. * (stats.norm.cdf(height_of_sample / 2. / beam_sd) - 0.5)
    return probability


def beamfrackernel(kernelx, kernely, length, angle):
    '''
    return the beam fraction intercepted by a sample of length length at sample
    tilt angle.
    The beam has the shape 'kernel', a 2 row array, which gives the PDF for the
    beam intensity as a function of height. The first row is position, the
    second row is probability at that position.
    '''
    height_of_sample = length * np.sin(np.radians(angle))
    total = integrate.simps(kernely, kernelx)
    lowlimit = np.where(-height_of_sample / 2. >= kernelx)[0][-1]
    hilimit = np.where(height_of_sample / 2. <= kernelx)[0][0]

    area = integrate.simps(kernely[lowlimit: hilimit + 1], kernelx[lowlimit: hilimit + 1])
    return area / total


def height_of_beam_after_dx(d1, d2, L12, distance):
    """
    Calculate the widths of beam a given distance away from a collimation slit.

    if distance >= 0, then it's taken to be the distance after d2.
    if distance < 0, then it's taken to be the distance before d1.

    Parameters:
        d1 - opening of first collimation slit
        d2 - opening of second collimation slit
        L12 - distance between first and second collimation slits
        distance - distance from first or last slit to a given position
    Units - equivalent distances (inches, mm, light years)

    Returns:
        (umbra, penumbra)

    """

    alpha = (d1 + d2) / 2. / L12
    beta = abs(d1 - d2) / 2. / L12
    if distance >= 0:
        return (beta * distance * 2) + d2, (alpha * distance * 2) + d2
    else:
        return (beta * abs(distance) * 2) + d1, (alpha * abs(distance) * 2) + d1


def actual_footprint(d1, d2, L12, L2S, angle):
    '''
    Calculate the actual footprint on a sample.
    Parameters:
        d1 - opening of first collimation slit
        d2 - opening of second collimation slit
        L12 - distance between first and second collimation slits
        L2S - distance from second collimation slit to sample

    Returns:
        (umbra_footprint, penumbra_footprint)

    '''
    umbra, penumbra = height_of_beam_after_dx(d1, d2, L12, L2S)
    return  umbra / np.radians(angle), penumbra / np.radians(angle)


def slit_optimiser(footprint,
                  resolution,
                  angle = 1.,
                  L12 = 2859.5,
                  L2S = 180,
                  LS3 = 290.5,
                  LSD = 2500,
                  verbose = True):
    """
    Optimise slit settings for a given angular resolution, and a given footprint.

    footprint: float
        maximum footprint onto sample (mm)
    resolution: float
        fractional dtheta/theta resolution (FWHM)
    angle: float, optional
        angle of incidence in degrees
    """
    if verbose:
        print('_____________________________________________')
        print('FOOTPRINT calculator - Andrew Nelson 2013')
        print('INPUT')
        print('footprint:', footprint, 'mm')
        print('fractional angular resolution (FWHM):', resolution)
        print('theta:', angle, 'degrees')

    d1star = lambda d2star : np.sqrt(1 - np.power(d2star, 2))
    L1star = 0.68 * footprint/L12/resolution

    gseekfun = lambda d2star : np.power((d2star + L2S / L12 * (d2star + d1star(d2star))) - L1star, 2)

    res = optimize.minimize_scalar(gseekfun, method='bounded', bounds=(0, 1))
    if res['success'] is False:
        print('ERROR: Couldnt find optimal solution, sorry')
        return None

    optimal_d2star = res['x']
    optimal_d1star = d1star(optimal_d2star)
    if optimal_d2star > optimal_d1star:
        # you found a minimum, but this may not be the optimum size of the slits.
        multfactor = 1
        optimal_d2star = 1/np.sqrt(2)
        optimal_d1star = 1/np.sqrt(2)
    else:
        multfactor = optimal_d2star / optimal_d1star

    d1 = optimal_d1star * resolution / 0.68 * np.radians(angle) * L12
    d2 = d1 * multfactor

    tmp, height_at_S4 = height_of_beam_after_dx(d1, d2, L12, L2S + LS3)
    tmp, height_at_detector = height_of_beam_after_dx(d1, d2, L12, L2S + LSD)
    tmp, _actual_footprint = actual_footprint(d1, d2, L12, L2S, angle)

    if verbose:
        print('OUTPUT')
        if multfactor == 1:
            print('Your desired resolution results in a smaller footprint than the sample supports.')
            suggested_resolution =  resolution * footprint / _actual_footprint
            print('You can increase flux using a resolution of', suggested_resolution, 'and still keep the same footprint.')
        print('d1', d1, 'mm')
        print('d2', d2, 'mm')
        print('footprint:', _actual_footprint, 'mm')
        print('height at S4:', height_at_S4, 'mm')
        print('height at detector:', height_at_detector, 'mm')
        print('[d2star', optimal_d2star, ']')
        print('_____________________________________________')

    return d1, d2
