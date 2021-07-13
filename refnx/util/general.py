#!/usr/bin/python
"""
Functions for various calculations related to reflectometry
"""
import numpy as np
from scipy import stats, integrate, constants, optimize

# h / m = 3956
K = constants.h / constants.m_n * 1.0e10


def div(d1, d2, L12=2859):
    """
    Calculate the angular resolution for a set of collimation conditions

    Parameters
    ----------
    d1 : float
        slit 1 opening
    d2 : float
        slit 2 opening
    L12 : float
        distance between slits

    Returns
    -------
    dtheta, alpha, beta
        dtheta is the FWHM of the Gaussian approximation to the trapezoidal
        resolution function
        alpha is the angular divergence of the penumbra
        beta is the angular divergence of the umbra

    When calculating dtheta / theta values for resolution, then dtheta is the
    value you need to use.
    See equations 11-14 in [1]_.

    .. [1] de Haan, V.-O.; de Blois, J.; van der Ende, P.; Fredrikze, H.;
    van der Graaf, A.; Schipper, M.; van Well, A. A. & J., v. d. Z. ROG, the
    neutron reflectometer at IRI Delft Nuclear Instruments and Methods in
    Physics Research A, 1995, 362, 434-453
    """
    divergence = 0.68 * 0.68 * (d1 ** 2 + d2 ** 2) / (L12 ** 2)
    alpha = (d1 + d2) / 2.0 / L12
    beta = abs(d1 - d2) / 2.0 / L12
    return np.degrees(np.sqrt(divergence)), np.degrees(alpha), np.degrees(beta)


def q(angle, wavelength):
    """
    Calculate Q given angle of incidence and wavelength

    Parameters
    ----------
    angle: float
        angle of incidence (degrees)
    wavelength: float
        wavelength of radiation (Angstrom)

    Returns
    -------
    Q: float
        Momentum transfer (A**-1)
    """
    return 4 * np.pi * np.sin(np.radians(angle)) / wavelength


def q2(omega, twotheta, phi, wavelength):
    """
    Convert angles and wavelength (lambda) to Q vector.

    Parameters
    ----------
    omega: float
        angle of incidence of beam (with respect to xy plane).
    twotheta: float
        angle between direct beam and the projection of the reflected beam onto
        xz plane.
    phi: float
        azimuthal angle between reflected beam and xz plane.
    wavelength: float

    Returns
    -------
    Qx, Qy, Qz : float, float, float
        Momentum transfer.

    Notes
    -----
    All angles are assumed to be in degrees.

    coordinate system:
    The beam is incident in the xz plane.
    x - along beam direction (in small angle approximation)
    y - transverse to beam direction, in plane of sample
    z - normal to sample plane.

    The xy plane is equivalent to the sample plane.
    """
    omega = np.radians(omega)
    twotheta = np.radians(twotheta)
    phi = np.radians(phi)

    qx = (
        2
        * np.pi
        / wavelength
        * (np.cos(twotheta - omega) * np.cos(phi) - np.cos(omega))
    )
    qy = 2 * np.pi / wavelength * np.cos(twotheta - omega) * np.sin(phi)
    qz = 2 * np.pi / wavelength * (np.sin(twotheta - omega) + np.sin(omega))

    return qx, qy, qz


def wavelength(q, angle):
    """
    calculate wavelength given Q vector and angle

    Parameters
    ----------
    q : float
        Wavevector (A**-1)
    angle : float
        angle of incidence (degrees)

    Returns
    -------
    wavelength : float
        Wavelength of radiation (A)
    """
    return 4.0 * np.pi * np.sin(angle * np.pi / 180.0) / q


def angle(q, wavelength):
    """
    calculate angle given Q and wavelength

    Parameters
    ----------
    q : float
        Wavevector (A**-1)
    wavelength : float
        Wavelength of radiation (A)

    Returns
    -------
    angle : float
        angle of incidence (degrees)
    """
    return np.arcsin((q * wavelength) / 4.0 / np.pi) * 180 / np.pi


def qcrit(SLD1, SLD2):
    """
    calculate critical Q vector given SLD of super and subphases

    Parameters
    ----------
    SLD1: float
        SLD of superphase (10^-6 A^-2)
    SLD2: float
        SLD of subphase (10^-6 A^-2)

    Returns
    -------
    qc: float
        Critical Q vector for a reflectivity system.
    """
    return np.sqrt(16.0 * np.pi * (SLD2 - SLD1) * 1.0e-6)


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

    Returns
    -------
    tauC: float
        The burst time of a double chopper pair (s)

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
    r"""
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


def velocity_wavelength(velocity):
    r"""
    Converts neutron velocity to wavelength

    Parameters
    ----------
    velocity: float
        velocity of neutron in m/s

    Returns
    -------
    wavelength: float
        wavelength of neutron in A
    """
    return K / velocity


def wavelength_energy(wavelength):
    r"""
    Converts wavelength to energy in meV

    Parameters
    ----------
    wavelength : float
        Wavelength in Angstrom.

    Returns
    -------
    energy : float
        Energy in meV.

    """
    c = 0.5e23 / constants.eV * constants.h ** 2 / constants.m_n
    return c / wavelength ** 2


def energy_wavelength(energy):
    r"""
    Converts wavelength to energy in meV

    Parameters
    ----------
    energy : float
        Energy in meV.

    Returns
    -------
    wavelength : float
        Wavelength in Angstrom.
    """
    c = 0.5e23 / constants.eV * constants.h ** 2 / constants.m_n
    return np.sqrt(c / energy)


def double_chopper_frequency(min_wavelength, max_wavelength, L, N=1):
    r"""
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

    Returns
    -------
    max_freq: float
        The maximum frequency a double chopper system can use to avoid frame
        overlap.
    """
    return K / ((max_wavelength - min_wavelength) * L * N)


def resolution_double_chopper(
    wavelength, z0=0.358, R=0.35, freq=24, H=0.005, xsi=0, L=7.5, tau_da=0
):
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

    Returns
    -------
    res: float
        Fractional wavelength resolution of a double chopper system.
    """
    TOF = L / wavelength_velocity(wavelength)
    tc = tauC(wavelength, xsi=xsi, z0=z0, freq=freq)
    tau_h = H / R / (2 * np.pi * freq)
    return 0.68 * np.sqrt(
        (tc / TOF) ** 2 + (tau_h / TOF) ** 2 + (tau_da / TOF) ** 2
    )


def resolution_single_chopper(
    wavelength, R=0.35, freq=24, H=0.005, phi=60, L=7.5
):
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

    Returns
    -------
    transmission: float
        Transmission of a single chopper system.
    """
    TOF = L / wavelength_velocity(wavelength)
    tauH = H / R / (2.0 * np.pi * freq)
    tauC = np.radians(phi) / (2.0 * np.pi * freq)
    return 0.68 * np.sqrt((tauC / TOF) ** 2 + (tauH / TOF) ** 2)


def transmission_double_chopper(
    wavelength, z0=0.358, R=0.35, freq=24, N=1, H=0.005, xsi=0
):
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

    Returns
    -------
    transmission: float
        The transmission of a double chopper system.

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


def _neutron_transmission_depth(material, wavelength):
    """
    Calculate the penetration depth for a material with a given neutron
    wavelength

    Parameters
    ----------
    material : pt.Formula

    wavelength : float
        neutron wavelength in Angstrom
    """
    import periodictable as pt

    return 10.0 * pt.neutron_scattering(material, wavelength=wavelength)[-1]


def neutron_transmission(formula, density, wavelength, thickness):
    """
    Calculates the transmission of neutrons through a material.
    Includes absorption + scattering (coherent+incoherent) cross sections.

    Parameters
    ----------
    formula : str
        Chemical formula of the material.
    density : float
        material density in g/cm^3
    wavelength : float, np.ndarray
        wavelength of neutron in Angstrom
    thickness : float
        thickness of material in mm

    Returns
    -------
    transmission : float or np.ndarray
        transmission of material
    """
    import periodictable as pt

    material = pt.formula(formula, density=density)

    _depth_fn = np.vectorize(_neutron_transmission_depth, excluded={0})
    depths = _depth_fn(material, wavelength)
    transmission = np.exp(-(thickness / depths))
    return transmission


def xray_wavelength(energy):
    """
    convert energy (keV) to wavelength (angstrom)
    """
    return 12.398 / energy


def xray_energy(wavelength):
    """
    convert energy (keV) to wavelength (angstrom)
    """
    return 12.398 / wavelength


def penetration_depth(qq, rho):
    """
    Calculates the penetration depth of a neutron/xray beam

    Parameters
    ----------
    qq: float
        Q values to calculate the penetration depth at
    rho: float or complex
        Complex SLD of material

    Returns
    -------
    penetration_depth: float
    """
    kk = 0.25 * qq ** 2.0
    kk -= 4 * np.pi * rho
    temp = np.sqrt(kk + 0j)
    return np.abs(1 / temp.imag)


def beamfrac(FWHM, length, angle):
    """
    Calculate the beam fraction intercepted by a sample.

    Parameters
    ----------
    FWHM: float
        The FWHM of the beam height
    length: float
        Length of the sample in mm
    angle: float
        Angle that the sample makes w.r.t the beam (degrees)

    Returns
    -------
    beamfrac: float
        The fraction of the beam that intercepts the sample
    """
    height_of_sample = length * np.sin(np.radians(angle))
    beam_sd = FWHM / 2 / np.sqrt(2 * np.log(2))
    probability = 2.0 * (
        stats.norm.cdf(height_of_sample / 2.0 / beam_sd) - 0.5
    )
    return probability


def beamfrackernel(kernelx, kernely, length, angle):
    """
    The beam fraction intercepted by a sample, used for calculating footprints.

    Parameters
    ----------
    kernelx: array-like
        x axis for the probability kernel
    kernely: array-like
        probability kernel describing the intensity distribution of the beam
    length: float
        length of the sample
    angle: float
        angle of incidence (degrees)

    Returns
    -------
    fraction: float
        The fraction of the beam intercepted by a sample.
    """
    height_of_sample = length * np.sin(np.radians(angle))
    total = integrate.simps(kernely, kernelx)
    lowlimit = np.where(-height_of_sample / 2.0 >= kernelx)[0][-1]
    hilimit = np.where(height_of_sample / 2.0 <= kernelx)[0][0]

    area = integrate.simps(
        kernely[lowlimit : hilimit + 1], kernelx[lowlimit : hilimit + 1]
    )
    return area / total


def height_of_beam_after_dx(d1, d2, L12, distance):
    """
    Calculate the total widths of beam a given distance away from a collimation
    slit.

    if distance >= 0, then it's taken to be the distance after d2.
    if distance < 0, then it's taken to be the distance before d1.

    Parameters
    ----------
    d1: float
        opening of first collimation slit
    d2: float
        opening of second collimation slit
    L12: float
        distance between first and second collimation slits
    distance: float
        distance from first or last slit to a given position

    Notes
    -----
    Units - equivalent distances (inches, mm, light years)

    Returns
    -------
    (umbra, penumbra): float, float
        full width of umbra and penumbra

    """

    alpha = (d1 + d2) / 2.0 / L12
    beta = abs(d1 - d2) / 2.0 / L12
    if distance >= 0:
        return (beta * distance * 2) + d2, (alpha * distance * 2) + d2
    else:
        return (
            (beta * abs(distance) * 2) + d1,
            (alpha * abs(distance) * 2) + d1,
        )


def actual_footprint(d1, d2, L12, L2S, angle):
    """
    Calculate the actual footprint on a reflectivity sample.

    Parameters
    ----------
    d1: float
        opening of first collimation slit
    d2: float
        opening of second collimation slit
    L12: float
        distance between first and second collimation slits
    L2S: float
        distance from second collimation slit to sample
    angle: float
        angle of incidence of sample (degrees)

    Returns
    -------
    (umbra_footprint, penumbra_footprint): float, float
        Footprint of the umbra and penumbra

    """
    umbra, penumbra = height_of_beam_after_dx(d1, d2, L12, L2S)
    return umbra / np.radians(angle), penumbra / np.radians(angle)


def slit_optimiser(
    footprint,
    resolution,
    angle=1.0,
    L12=2859.5,
    L2S=180,
    LS3=290.5,
    LSD=2500,
    verbose=True,
):
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
    if verbose:
        print("_____________________________________________")
        print("FOOTPRINT calculator - Andrew Nelson 2013")
        print("INPUT")
        print("footprint:", footprint, "mm")
        print("fractional angular resolution (FWHM):", resolution)
        print("theta:", angle, "degrees")

    def d1star(d2star):
        return np.sqrt(1 - np.power(d2star, 2))

    L1star = 0.68 * footprint / L12 / resolution

    def gseekfun(d2star):
        return np.power(
            (d2star + L2S / L12 * (d2star + d1star(d2star))) - L1star, 2
        )

    res = optimize.minimize_scalar(gseekfun, method="bounded", bounds=(0, 1))
    if res["success"] is False:
        print("ERROR: Couldnt find optimal solution, sorry")
        return None

    optimal_d2star = res["x"]
    optimal_d1star = d1star(optimal_d2star)
    if optimal_d2star > optimal_d1star:
        # you found a minimum, but this may not be the optimum size of the
        # slits.
        multfactor = 1
        optimal_d2star = 1 / np.sqrt(2)
        optimal_d1star = 1 / np.sqrt(2)
    else:
        multfactor = optimal_d2star / optimal_d1star

    d1 = optimal_d1star * resolution / 0.68 * np.radians(angle) * L12
    d2 = d1 * multfactor

    tmp, height_at_S4 = height_of_beam_after_dx(d1, d2, L12, L2S + LS3)
    tmp, height_at_detector = height_of_beam_after_dx(d1, d2, L12, L2S + LSD)
    tmp, _actual_footprint = actual_footprint(d1, d2, L12, L2S, angle)

    if verbose:
        print("OUTPUT")
        if multfactor == 1:
            print(
                "Your desired resolution results in a smaller footprint than"
                " the sample supports."
            )
            suggested_resolution = resolution * footprint / _actual_footprint
            print(
                "You can increase flux using a resolution of",
                suggested_resolution,
                "and still keep the same footprint.",
            )
        print("d1", d1, "mm")
        print("d2", d2, "mm")
        print("footprint:", _actual_footprint, "mm")
        print("height at S4:", height_at_S4, "mm")
        print("height at detector:", height_at_detector, "mm")
        print("[d2star", optimal_d2star, "]")
        print("_____________________________________________")

    return d1, d2


def _dict_compare(d1, d2):
    """
    Rudimentary check to see if two dict are the same. This won't do recursive
    checking (e.g. if items in d1 or d2 are dicts)

    Parameters
    ----------
    d1 : dict
    d2 : dict

    Returns
    -------
    truth : bool
        Are two dicts the same
    """
    if len(d1) != len(d2):
        return False

    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    if len(intersect_keys) != len(d1):
        return False

    for o in intersect_keys:
        if isinstance(d1[o], np.ndarray) and isinstance(d2[o], np.ndarray):
            # both numpy arrays
            if not np.array_equal(d1[o], d2[o]):
                return False
            continue

        if not isinstance(d1[o], d2[o].__class__) or d1[o] != d2[o]:
            return False

    return True


def _dict_compare_keys(d1, d2, *keys):
    """
    Check to see if specific key value pairs are the same within
    two different dictionaries.

    Parameters
    ----------
    d1      :   dict
    d2      :   dict
    keys    :   list of str
        list of keys to check are equal between the dicts

    Returns
    -------
    True if the key-value pairs are the same between `d1` and `d2`,
    otherwise False.
    """
    for k in keys:
        if isinstance(d1[k], np.ndarray) and isinstance(d2[k], np.ndarray):
            # both numpy arrays
            if not np.array_equal(d1[k], d2[k]):
                return False
            continue

        if not isinstance(d1[k], d2[k].__class__) or d1[k] != d2[k]:
            return False

    return True
