"""
*Calculates the specular (Neutron or X-ray) reflectivity from a stratified
series of layers.

The refnx code is distributed under the following license:

Copyright (c) 2015 A. R. J. Nelson, ANSTO

Permission to use and redistribute the source code or binary forms of this
software and its documentation, with or without modification is hereby
granted provided that the above notice of copyright, these terms of use,
and the disclaimer of warranty below appear in the source code and
documentation, and that none of the names of above institutions or
authors appear in advertising or endorsement of works derived from this
software without specific prior written permission from all parties.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THIS SOFTWARE.

"""

from functools import reduce

import numpy as np
import jax.numpy as jnp
from jax import jit
from refnx.reflect.reflect_model import gauss_legendre

TINY = 1e-30

_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5


def jabeles(q, layers, scale=1.0, bkg=0, threads=0):
    # Note, this code must have 64 bit operation enabled before it is used
    # You can do this with:
    # from jax import config
    # config.update("jax_enable_x64", True)

    qvals = q.astype(jnp.float64)
    flatq = qvals.ravel()

    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    mi00 = jnp.ones((npnts, nlayers + 1), jnp.complex128)

    sld = jnp.zeros(nlayers + 2, jnp.complex128)

    # addition of TINY is to ensure the correct branch cut
    # in the complex sqrt calculation of kn.
    sld = sld.at[1:].add(
        ((layers[1:, 1] - layers[0, 1]) + 1j * (jnp.abs(layers[1:, 2]) + TINY))
        * 1.0e-6
    )

    kn = jnp.sqrt(flatq[:, jnp.newaxis] ** 2.0 / 4.0 - 4.0 * jnp.pi * sld)
    # reflectances for each layer
    # rj.shape = (npnts, nlayers + 1)
    damping = jnp.exp(-2.0 * kn[:, :-1] * kn[:, 1:] * layers[1:, 3] ** 2)
    rj = (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:]) * damping

    # characteristic matrices for each layer
    # miNN.shape = (npnts, nlayers + 1)
    if nlayers:
        mi00 = mi00.at[:, 1:].set(
            jnp.exp(kn[:, 1:-1] * 1j * jnp.fabs(layers[1:-1, 0]))
        )
    mi11 = 1.0 / mi00
    mi10 = rj * mi00
    mi01 = rj * mi11

    # initialise matrix total
    mrtot00 = mi00[:, 0]
    mrtot01 = mi01[:, 0]
    mrtot10 = mi10[:, 0]
    mrtot11 = mi11[:, 0]

    for _mi00, _mi10, _mi01, _mi11 in zip(
        mi00[:, 1:].T, mi10[:, 1:].T, mi01[:, 1:].T, mi11[:, 1:].T
    ):
        # matrix multiply mrtot by characteristic matrix
        p00 = mrtot00 * _mi00 + mrtot10 * _mi01
        p10 = mrtot00 * _mi10 + mrtot10 * _mi11
        p01 = mrtot01 * _mi00 + mrtot11 * _mi01
        p11 = mrtot01 * _mi10 + mrtot11 * _mi11

        mrtot00, mrtot01, mrtot10, mrtot11 = p00, p01, p10, p11

    r = mrtot01 / mrtot00
    reflectivity = r * jnp.conj(r)

    return scale * jnp.real(jnp.reshape(reflectivity, qvals.shape)) + bkg


# abeles_jax = jabeles
abeles_jax = jit(jabeles)


def jax_smeared_kernel_pointwise(
    q: jnp.ndarray,
    dqvals: jnp.ndarray,
    layers: jnp.ndarray,
    scale: float | jnp.ndarray = 1.0,
    bkg: float | jnp.ndarray = 0.0,
    quad_order: int = 17,
) -> jnp.ndarray:
    """
    Resolution-smeared reflectivity using Gauss-Legendre quadrature.
    Fully JAX-traceable — gradients flow through ``layers``, ``scale``,
    and ``bkg``.

    This replaces ``jax_smeared_kernel_pointwise``, which uses numpy
    internally and is therefore not differentiable.  The maths are
    identical; only the array library changes.

    Parameters
    ----------
    q : (N,) jnp.ndarray
        Nominal Q values (Å⁻¹).
    dqvals : (N,) jnp.ndarray
        Per-point dQ resolution
    layers : (M, 4) jnp.ndarray
        Layer stack as returned by ``params_to_slabs``.
    scale, bkg :
        Passed directly to ``jabeles``.
    quad_order : int
        Number of Gauss-Legendre nodes.  17 matches the default used by
        ``jax_smeared_kernel_pointwise`` and ``ReflectModel``.

    Returns
    -------
    smeared_r : (N,) jnp.ndarray
        Smeared reflectivity values.

    Notes
    -----
    The integral is

        R_smeared(q) = ∫ G(q' | q, sigma) R(q') dq'

    where G is a Gaussian with sigma = dQ / (2 sqrt(2 ln 2)), and the
    integration limits are ±3.5 sigma (matching ``_INTLIMIT`` in
    ``_jax_reflect.py``).  We use a change of variables to map each
    per-point integral onto [-1, 1] for Gauss-Legendre.
    """
    _FWHM = 2.0 * jnp.sqrt(2.0 * jnp.log(2.0))
    _INTLIMIT = 3.5

    # Gauss-Legendre nodes and weights on [-1, 1] — computed once in numpy
    # (pure constants, not part of the JAX trace).
    abscissa, weights = gauss_legendre(quad_order)  # numpy arrays
    abscissa_j = jnp.array(abscissa, dtype=jnp.float64)  # (P,)
    weights_j = jnp.array(weights, dtype=jnp.float64)  # (P,)

    # Gaussian kernel evaluated at the abscissae, also a constant.
    prefactor = 1.0 / jnp.sqrt(2.0 * jnp.pi)
    gaussvals = prefactor * jnp.exp(-0.5 * abscissa_j**2)  # (P,)
    gw = gaussvals * weights_j  # (P,)

    # Integration limits for each Q point: [q - 3.5*sigma, q + 3.5*sigma]
    # where sigma = dq_abs / FWHM.
    va = q - _INTLIMIT * dqvals / _FWHM  # (N,)
    vb = q + _INTLIMIT * dqvals / _FWHM  # (N,)

    # Map GL nodes onto each per-Q interval.
    # q_grid[i, p] = q value for Q-point i at GL node p.
    # Shape: (N, P)
    q_grid = (
        abscissa_j[jnp.newaxis, :] * (vb - va)[:, jnp.newaxis]
        + (vb + va)[:, jnp.newaxis]
    ) / 2.0

    # Evaluate unsmeared reflectivity on the full (N, P) Q grid.
    # jabeles handles arbitrary-shaped q input by ravelling internally.
    r_grid = jabeles(q_grid, layers, scale=scale, bkg=bkg)  # (N, P)

    # Gaussian-weighted sum over quadrature nodes, scaled by half-width.
    smeared = jnp.sum(r_grid * gw[jnp.newaxis, :], axis=1) * _INTLIMIT

    return smeared
