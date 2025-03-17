import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from refnx.reflect import reflectivity


def res(qq, layer, resolution=5):
    resolution /= 100
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    def gauss(x, s):
        return 1.0 / s / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2 / s / s)

    lowQ = np.min(qq)
    highQ = np.max(qq)
    if lowQ <= 0:
        lowQ = 1e-6

    start = np.log10(lowQ) - 6 * resolution / 2.35482
    finish = np.log10(highQ * (1 + 6 * resolution / 2.35482))
    interpnum = np.round(
        np.abs(
            1
            * (np.abs(start - finish))
            / (1.7 * resolution / 2.35482 / gaussgpoint)
        )
    )
    xtemp = np.linspace(start, finish, int(interpnum))

    gauss_x = np.linspace(-1.7 * resolution, 1.7 * resolution, gaussnum)
    gauss_y = gauss(gauss_x, resolution / (2 * np.sqrt(2 * np.log(2))))

    rvals = reflectivity(np.power(10, xtemp), layer)
    smeared_rvals = fftconvolve(rvals, gauss_y, mode="same")
    interpolator = interp1d(np.power(10, xtemp), smeared_rvals)

    smeared_output = interpolator(qq)
    smeared_output /= np.sum(gauss_y)
    return smeared_output
