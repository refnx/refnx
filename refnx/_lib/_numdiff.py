"""numerical differentiation function, gradient, Jacobian, and Hessian
Author : josef-pkt
License : BSD
Notes
-----
These are simple forward differentiation, so that we have them available
without dependencies.
* Jacobian should be faster than numdifftools because it doesn't use loop over
  observations.
* numerical precision will vary and depend on the choice of stepsizes

VENDORED Aug2017
"""

import numpy as np


# NOTE: we only do double precision internally so far
EPS = np.finfo(np.float64).eps


def _get_epsilon(x, s, epsilon, n):
    if epsilon is None:
        h = EPS ** (1.0 / s) * np.maximum(np.abs(x), 0.1)
    else:
        if np.isscalar(epsilon):
            h = np.empty(n)
            h.fill(epsilon)
        else:  # pragma : no cover
            h = np.asarray(epsilon)
            if h.shape != x.shape:
                raise ValueError(
                    "If h is not a scalar it must have the same shape as x."
                )
    return h


def approx_hess2(x, f, epsilon=None, args=(), kwargs={}, return_grad=False):
    """
    Calculate Hessian with finite difference derivative approximation
    Parameters
    ----------
    x : array_like
       value at which function derivative is evaluated
    f : function
       function of one array f(x, `*args`, `**kwargs`)
    epsilon : float or array-like, optional
       Stepsize used, if None, then stepsize is automatically chosen
       according to EPS**(1/3)*x.
    args : tuple
        Arguments for function `f`.
    kwargs : dict
        Keyword arguments for function `f`.

    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian

    Notes
    -----
    Equation (%(equation_number)s) in Ridout. Computes the Hessian as::
      %(equation)s
    where e[j] is a vector with element j == 1 and the rest are zero and
    d[i] is epsilon[i].
    References
    ----------:
    Ridout, M.S. (2009) Statistical applications of the complex-step method
        of numerical differentiation. The American Statistician, 63, 66-74
    """
    n = len(x)
    # NOTE: ridout suggesting using eps**(1/4)*theta
    h = _get_epsilon(x, 3, epsilon, n)
    ee = np.diag(h)
    f0 = f(*((x,) + args), **kwargs)
    # Compute forward step
    g = np.zeros(n)
    gg = np.zeros(n)
    for i in range(n):
        g[i] = f(*((x + ee[i, :],) + args), **kwargs)
        gg[i] = f(*((x - ee[i, :],) + args), **kwargs)

    hess = np.outer(h, h)  # this is now epsilon**2
    # Compute "double" forward step
    for i in range(n):
        for j in range(i, n):
            hess[i, j] = (
                f(*((x + ee[i, :] + ee[j, :],) + args), **kwargs)
                - g[i]
                - g[j]
                + f0
                + f(*((x - ee[i, :] - ee[j, :],) + args), **kwargs)
                - gg[i]
                - gg[j]
                + f0
            ) / (2 * hess[i, j])
            hess[j, i] = hess[i, j]
    if return_grad:
        grad = (g - f0) / h
        return hess, grad
    else:
        return hess
