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
    mi10 = rj * mi11
    mi01 = rj * mi00

    mi = jnp.zeros((npnts, nlayers + 1, 2, 2), jnp.complex128)

    mi = mi.at[:, :, 0, 0].set(mi00)
    mi = mi.at[:, :, 0, 1].set(mi01)
    mi = mi.at[:, :, 1, 1].set(mi11)
    mi = mi.at[:, :, 1, 0].set(mi10)

    sub = [jnp.squeeze(v) for v in jnp.hsplit(mi, nlayers + 1)]
    mrtot = reduce(jnp.matmul, sub[1:], sub[0])

    r = mrtot[:, 1, 0] / mrtot[:, 0, 0]
    reflectivity = r * jnp.conj(r) * scale
    reflectivity = reflectivity.at[...].add(bkg)

    return jnp.real(jnp.reshape(reflectivity, qvals.shape))


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
