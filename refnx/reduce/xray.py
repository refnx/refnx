import numpy as np
import refnx.util.general as general
import refnx.util.ErrorProp as EP
import xml.etree.ElementTree as et
from refnx.dataset import ReflectDataset

# mm
XRR_BEAMWIDTH_SD = 0.019449


def reduce_xrdml(f, bkg=None, scale=None, sample_length=None):
    """
    Reduces a Panalytical XRDML file

    Parameters
    ----------
    f: file-like object or string
        The specular reflectivity (XRDML) file of interest
    bkg: list
        A list of file-like objects or strings that contain background
        measurements. The background is assumed to have the same number of
        points as the specular reflectivity curve.  The backgrounds are
        averaged and subtracted from the specular reflectivity
    scale: float, None
        The direct beam intensity (cps). If `scale is None` then the dataset
        is scaled by the point with maximum intensity below Q = 0.0318 (Q_crit
        for Si at 8.048 keV).
    sample_length: None or float
        If None then no footprint correction is done. Otherwise the transverse
        footprint of the sample (mm).

    Returns
    -------
    dataset: refnx.dataset.ReflectDataset
        The specular reflectivity as a function of momentum transfer, Q.
    """

    spec = parse_xrdml_file(f)

    reflectivity = spec["intensities"] / spec["count_time"]
    reflectivity_s = np.sqrt(spec["intensities"]) / spec["count_time"]

    # do the background subtraction
    if bkg is not None:
        bkgds = [parse_xrdml_file(fi) for fi in bkg]

        bkgd_refs = np.r_[[bkgd["intensities"] for bkgd in bkgds]]
        bkgd_refs_s = np.r_[
            [
                np.sqrt(bkgd["intensities"]) / bkgd["count_time"]
                for bkgd in bkgds
            ]
        ]
        bkgd_refs_var = bkgd_refs_s ** 2
        weights = 1.0 / bkgd_refs_var
        numerator = np.sum(bkgd_refs * weights, axis=0)
        denominator = np.sum(weights, axis=0)

        total_bkgd = numerator / denominator
        total_bkgd_s = np.sqrt(1 / denominator)

        reflectivity, reflectivity_s = EP.EPsub(
            reflectivity, reflectivity_s, total_bkgd, total_bkgd_s
        )

    # work out the Q values
    qx, qy, qz = general.q2(
        spec["omega"],
        spec["twotheta"],
        np.zeros_like(spec["omega"]),
        spec["wavelength"],
    )

    # do a footprint correction
    if sample_length is not None:
        footprint_correction = general.beamfrac(
            np.array([XRR_BEAMWIDTH_SD]) * 2.35,
            np.array([sample_length]),
            spec["omega"],
        )
        reflectivity /= footprint_correction
        reflectivity_s /= footprint_correction

    # divide by the direct beam intensity
    # assumes that the direct beam intensity is enormous, so the counting
    # uncertainties in the scale factor are negligible.
    if scale is None:
        # no scale factor was specifed, so normalise by highest intensity point
        # below Qc for Silicon at 8.048 keV
        below_qc = qz[qz < 0.0318]
        if len(below_qc):
            scale = np.max(reflectivity[qz < 0.0318])

    reflectivity /= scale
    reflectivity_s /= scale

    d = ReflectDataset(data=(qz, reflectivity, reflectivity_s))

    return d


def parse_xrdml_file(f):
    """
    Parses an XRML file

    Parameters
    ----------
    f: file-like object or string

    Returns
    -------
    d: dict
        A dictionary containing the XRDML file information.  The following keys
        are used:

        'intensities' - np.ndarray
            Intensities
        'twotheta' - np.ndarray
            Two theta values
        'omega' - np.ndarray
            Omega values
        'count_time' - float
            How long each point was counted for
        'wavelength' - float
            Wavelength of X-ray radiation
    """
    tree = et.parse(f)
    root = tree.getroot()
    ns = {"xrdml": "http://www.xrdml.com/XRDMeasurement/1.0"}

    query = {
        "intensities": ".//xrdml:intensities",
        "twotheta_start": ".//xrdml:positions[@axis='2Theta']"
        "/xrdml:startPosition",
        "twotheta_end": ".//xrdml:positions[@axis='2Theta']"
        "/xrdml:endPosition",
        "omega_start": ".//xrdml:positions[@axis='Omega']"
        "/xrdml:startPosition",
        "omega_end": ".//xrdml:positions[@axis='Omega']" "/xrdml:endPosition",
        "cnt_time": ".//xrdml:commonCountingTime",
        "kAlpha1": ".//xrdml:kAlpha1",
        "kAlpha2": ".//xrdml:kAlpha2",
        "ratio": ".//xrdml:ratioKAlpha2KAlpha1",
    }

    res = {key: root.find(value, ns).text for key, value in query.items()}

    kAlpha1 = float(res["kAlpha1"])
    kAlpha2 = float(res["kAlpha2"])
    ratio = float(res["ratio"])
    wavelength = (kAlpha1 + ratio * kAlpha2) / (1 + ratio)

    d = dict()

    intensities = np.fromstring(res["intensities"], sep=" ")
    n_pnts = intensities.size
    d["intensities"] = intensities
    d["twotheta"] = np.linspace(
        float(res["twotheta_start"]), float(res["twotheta_end"]), n_pnts
    )
    d["omega"] = np.linspace(
        float(res["omega_start"]), float(res["omega_end"]), n_pnts
    )
    d["count_time"] = float(res["cnt_time"])
    d["wavelength"] = wavelength

    return d


def process_offspec(f):
    """
    Process a 2D XRDML file and return qx, qz, intensity, dintensity

    Parameters
    ----------
    f: str or file-like

    Returns
    -------
    qx, qz, intensity, dintensity
    """

    x = et.parse(f)
    root = x.getroot()
    ns = {"xrdml": "http://www.xrdml.com/XRDMeasurement/1.0"}
    query = {
        "intensities": ".//xrdml:intensities",
        "twotheta_start": ".//xrdml:positions[@axis='2Theta']"
        "/xrdml:startPosition",
        "twotheta_end": ".//xrdml:positions[@axis='2Theta']"
        "/xrdml:endPosition",
        "omega_start": ".//xrdml:positions[@axis='Omega']"
        "/xrdml:startPosition",
        "omega_end": ".//xrdml:positions[@axis='Omega']" "/xrdml:endPosition",
        "cnt_time": ".//xrdml:commonCountingTime",
        "kAlpha1": ".//xrdml:kAlpha1",
        "kAlpha2": ".//xrdml:kAlpha2",
        "ratio": ".//xrdml:ratioKAlpha2KAlpha1",
    }

    res = {key: root.findall(value, ns) for key, value in query.items()}

    kAlpha1 = float(res["kAlpha1"][0].text)
    kAlpha2 = float(res["kAlpha2"][0].text)
    ratio = float(res["ratio"][0].text)
    wavelength = (kAlpha1 + ratio * kAlpha2) / (1 + ratio)

    intensity = [
        np.fromstring(ints.text, sep=" ") for ints in res["intensities"]
    ]
    twotheta_starts = np.array(
        [np.fromstring(ints.text, sep=" ") for ints in res["twotheta_start"]]
    )
    twotheta_ends = np.array(
        [np.fromstring(ints.text, sep=" ") for ints in res["twotheta_end"]]
    )
    omega_starts = np.array(
        [np.fromstring(ints.text, sep=" ") for ints in res["omega_start"]]
    )
    omega_ends = np.array(
        [np.fromstring(ints.text, sep=" ") for ints in res["omega_end"]]
    )
    cnt_time = np.array(
        [np.fromstring(ints.text, sep=" ") for ints in res["cnt_time"]]
    )

    intensity = np.array(intensity)
    dintensity = np.sqrt(intensity) / cnt_time
    intensity /= cnt_time

    omegas = []
    two_thetas = []

    for i in range(len(intensity)):
        omega = np.linspace(
            omega_starts[i], omega_ends[i], np.size(intensity, 1)
        )
        omegas.append(omega)
        two_theta = np.linspace(
            twotheta_starts[i], twotheta_ends[i], np.size(intensity, 1)
        )
        two_thetas.append(two_theta)

    omega = np.array(omegas)
    twotheta = np.array(two_thetas)
    qx, qy, qz = general.q2(omega, twotheta, 0, wavelength)

    return qx, qz, intensity, dintensity
