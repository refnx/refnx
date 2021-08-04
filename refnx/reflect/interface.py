"""
Interfacial models

Definitions taken from:

Svechnikov, M.; Pariev, D.; Nechay, A.; Salashchenko, N.; Chkhalo, N.;
Vainer, Y. & Gaman, D., "Extended model for the reconstruction of periodic
multilayers from extreme ultraviolet and X-ray reflectivity data",
Journal of Applied Crystallography, 2017, 50, 1428-1440

The Sinusoidal and Exponential definitions are incorrect in that paper
though. The correct equations are in:
Stearns, D. G. J. Appl. Phys., 1989, 65, 491–506.

The tanh definition has various definitions. This is taken from:
D. Bahr; W. Press; R. Jebasinski; S. Mantl, Phys. Rev. B, 1993, 47 (8), 4385
"""
import numpy as np
from scipy.stats import norm


_SQRT3 = np.sqrt(3.0)
_SQRT2 = np.sqrt(2.0)
_GAMMA = np.pi / np.sqrt(np.pi * np.pi - 8.0)


class Interface:
    """
    Defines an Interfacial profile
    """

    def __init__(self):
        pass

    def __call__(self, z, scale=1, loc=0):
        raise NotImplementedError(
            "You can't use the Interface superclass to calculate profiles"
        )


class Erf(Interface):
    """
    An Error function interfacial profile

    Notes
    -----
    Svechnikov, M.; Pariev, D.; Nechay, A.; Salashchenko, N.; Chkhalo, N.;
    Vainer, Y. & Gaman, D., "Extended model for the reconstruction of periodic
    multilayers from extreme ultraviolet and X-ray reflectivity data",
    Journal of Applied Crystallography, 2017, 50, 1428-1440
    """

    def __init__(self):
        super().__init__()

    def __call__(self, z, scale=1, loc=0):
        return norm.cdf(z, scale=scale, loc=loc)

    def __repr__(self):
        return "Erf()"


class Linear(Interface):
    """
    A Linear function interfacial profile

    Notes
    -----
    Stearns, D. G. J. Appl. Phys., 1989, 65, 491–506.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, z, scale=1, loc=0):
        new_z = z - loc
        f = 0.5 + new_z / (2 * _SQRT3 * scale)
        f[new_z <= -_SQRT3 * scale] = 0
        f[new_z >= _SQRT3 * scale] = 1
        return f

    def __repr__(self):
        return "Linear()"


class Exponential(Interface):
    """
    An Exponential interfacial profile

    Notes
    -----
    Stearns, D. G. J. Appl. Phys., 1989, 65, 491–506.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, z, scale=1, loc=0):
        new_z = z - loc
        f = np.zeros_like(new_z)
        f[new_z > 0] = 1 - 0.5 * np.exp(-_SQRT2 * new_z[new_z > 0] / scale)
        f[new_z <= 0] = 0.5 * np.exp(_SQRT2 * new_z[new_z <= 0] / scale)
        return f

    def __repr__(self):
        return "Exponential()"


class Tanh(Interface):
    """
    A hyperbolic tangent (tanh) interfacial profile

    Notes
    -----
    D. Bahr; W. Press; R. Jebasinski; S. Mantl,
    Phys. Rev. B,1993, 47 (8), 4385
    """

    def __init__(self):
        super().__init__()

    def __call__(self, z, scale=1, loc=0):
        arg = np.sqrt(2 / np.pi) * (z - loc) / scale
        return 0.5 * (1 + np.tanh(arg))

    def __repr__(self):
        return "Tanh()"


class Sinusoidal(Interface):
    """
    A sinusoidal (sin) interfacial profile

    Notes
    -----
    Stearns, D. G. J. Appl. Phys., 1989, 65, 491–506.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, z, scale=1, loc=0):
        new_z = z - loc
        f = 0.5 + 0.5 * np.sin(np.pi * new_z / _GAMMA / 2.0 / scale)
        f[new_z <= -_GAMMA * scale] = 0
        f[new_z >= _GAMMA * scale] = 1
        return f

    def __repr__(self):
        return "Sinusoidal()"


class Step(Interface):
    """
    A step function interfacial profile

    Notes
    -----
    Svechnikov, M.; Pariev, D.; Nechay, A.; Salashchenko, N.; Chkhalo, N.;
    Vainer, Y. & Gaman, D., "Extended model for the reconstruction of periodic
    multilayers from extreme ultraviolet and X-ray reflectivity data",
    Journal of Applied Crystallography, 2017, 50, 1428-1440
    """

    def __init__(self):
        super().__init__()

    def __call__(self, z, scale=1, loc=0):
        new_z = z - loc
        f = np.ones_like(new_z) * 0.5
        f[new_z <= -scale] = 0
        f[new_z >= scale] = 1
        return f

    def __repr__(self):
        return "Step()"
