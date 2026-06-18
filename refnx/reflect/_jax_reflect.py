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
from jax import jit, lax
from refnx.reflect.reflect_model import gauss_legendre

TINY = 1e-30

_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5


def jabeles(q, layers, scale=1.0, bkg=0, threads=0):
    qvals = q.astype(jnp.float64)
    flatq = qvals.ravel()

    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    # Build sld more directly (avoids mutable update)
    sld_vals = (
        (layers[1:, 1] - layers[0, 1]) + 1j * (jnp.abs(layers[1:, 2]) + TINY)
    ) * 1.0e-6
    sld = jnp.concatenate([jnp.zeros(1, jnp.complex128), sld_vals])

    kn = jnp.sqrt(flatq[:, jnp.newaxis] ** 2.0 / 4.0 - 4.0 * jnp.pi * sld)

    damping = jnp.exp(-2.0 * kn[:, :-1] * kn[:, 1:] * layers[1:, 3] ** 2)
    rj = (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:]) * damping

    # Compute mi00 for inner layers only, use exp(-x) instead of division
    if nlayers:
        exponent = kn[:, 1:-1] * 1j * jnp.fabs(layers[1:-1, 0])
        mi00_inner = jnp.exp(exponent)
        mi11_inner = jnp.exp(-exponent)

        # Full arrays including first layer (exponent=0 → mi00=1)
        mi00 = jnp.concatenate(
            [jnp.ones((npnts, 1), jnp.complex128), mi00_inner], axis=1
        )
        mi11 = jnp.concatenate(
            [jnp.ones((npnts, 1), jnp.complex128), mi11_inner], axis=1
        )
    else:
        mi00 = jnp.ones((npnts, 1), jnp.complex128)
        mi11 = mi00

    mi10 = rj * mi00
    mi01 = rj * mi11

    # Initialise with first layer
    mrtot00 = mi00[:, 0]
    mrtot01 = mi01[:, 0]
    mrtot10 = mi10[:, 0]
    mrtot11 = mi11[:, 0]

    # Use lax.scan instead of Python for-loop
    def body_fn(carry, x):
        m00, m01, m10, m11 = carry
        _mi00, _mi10, _mi01, _mi11 = x
        p00 = m00 * _mi00 + m10 * _mi01
        p10 = m00 * _mi10 + m10 * _mi11
        p01 = m01 * _mi00 + m11 * _mi01
        p11 = m01 * _mi10 + m11 * _mi11
        return (p00, p01, p10, p11), None

    if nlayers:
        # xs shape: each is (nlayers, npnts) — scan over layers axis
        xs = (
            mi00[:, 1:].T,
            mi10[:, 1:].T,
            mi01[:, 1:].T,
            mi11[:, 1:].T,
        )
        (mrtot00, mrtot01, mrtot10, mrtot11), _ = lax.scan(
            body_fn, (mrtot00, mrtot01, mrtot10, mrtot11), xs
        )

    r = mrtot01 / mrtot00
    reflectivity = r * jnp.conj(r)
    return scale * jnp.real(jnp.reshape(reflectivity, qvals.shape)) + bkg


abeles_jax = jit(jabeles)


# abeles_jax = jabeles
abeles_jax = jit(jabeles)


def jax_smeared_kernel_pointwise(qvals, w, dqvals, quad_order=17, threads=0):
    # get the gauss-legendre weights and abscissae
    abscissa, weights = gauss_legendre(quad_order)

    # get the normal distribution at that point
    prefactor = 1.0 / np.sqrt(2 * np.pi)

    def gauss(x):
        return np.exp(-0.5 * x * x)

    gaussvals = prefactor * gauss(abscissa * _INTLIMIT)

    # integration between -3.5 and 3.5 sigma
    va = qvals - _INTLIMIT * dqvals / _FWHM
    vb = qvals + _INTLIMIT * dqvals / _FWHM

    va = va[:, np.newaxis]
    vb = vb[:, np.newaxis]

    qvals_for_res = (np.atleast_2d(abscissa) * (vb - va) + vb + va) / 2.0
    smeared_rvals = abeles_jax(qvals_for_res, w)

    smeared_rvals = np.reshape(smeared_rvals, (qvals.size, abscissa.size))

    smeared_rvals *= np.atleast_2d(gaussvals * weights)
    return np.sum(smeared_rvals, 1) * _INTLIMIT
