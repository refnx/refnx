import numpy as np

"""
A simple propagation of uncertainty module.
# http://en.wikipedia.org/wiki/Propagation_of_uncertainty

For more sophisticated handling of uncertainties use the `uncertainties`
package.
"""


def EPadd(a, da, b, db, covar=0):
    """
    C = A + B
    """
    return (a + b), np.sqrt(da**2 + db**2 + 2 * covar)


def EPsub(a, da, b, db, covar=0):
    """
    C = A - B
    """
    return a - b, np.sqrt(da**2 + db**2 - 2 * covar)


def EPmul(a, da, b, db, covar=0):
    """
    C = A * B
    """
    return a * b, np.sqrt((b * da) ** 2 + (a * db) ** 2 + 2 * a * b * covar)


def EPmulk(a, da, k):
    """
    C = A * k
    """
    return a * k, np.absolute(da * k)


def EPdiv(a, da, b, db, covar=0):
    """
    C = A / B
    """
    return (
        a / b,
        np.sqrt(
            ((da / b) ** 2 + ((a**2) * (db**2) / (b**4)))
            - (2 * covar * a / (b**3))
        ),
    )


def EPpow(a, da, b, db, covar=0):
    """
    C = A ** B
    """
    c = a**b
    dc = np.absolute(c) * np.sqrt(
        ((da * b / a) ** 2)
        + (np.log(a) * db) ** 2
        + 2 * b * np.log(a) * covar / a
    )
    return c, dc


def EPpowk(a, da, k, n=1):
    """
    C = pow(k, A * n)
    """
    return (
        np.power(k, a * n),
        np.absolute(np.power(k, a * n) * n * da * np.log(k)),
    )


def EPlog(a, da, k=1, n=1):
    """
    C = n * log(k * A )
    """
    return n * np.log(k * a), np.absolute(n * da / a)


def EPlog10(a, da, k=1, n=1):
    """
    C = n * log10(k * A)
    """
    return n * np.log10(k * a), np.absolute(n * da / (a * np.log(10)))


def EPexp(a, da, k=1, n=1):
    """
    C = n * exp(k * A)
    """
    return n * np.exp(k * a), np.absolute(k * da * n * np.exp(k * a))


def EPsin(a, da):
    """
    C = sin (A)
    """
    return np.sin(a), np.absolute(np.cos(a) * da)


def EPcos(a, da):
    """
    C = cos (A)
    """
    return np.cos(a), np.absolute(-np.sin(a) * da)


def EPtan(a, da):
    """
    C = tan (A)
    """
    return np.tan(a), np.absolute(da / (np.cos(a) ** 2))
