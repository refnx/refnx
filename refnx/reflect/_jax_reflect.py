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

import jax.numpy as jnp
from jax import jit
from jax.ops import index, index_add, index_update

TINY = 1e-30


def jabeles(q, layers, scale=1.0, bkg=0, threads=0):
    qvals = q.astype(jnp.float64)
    flatq = qvals.ravel()

    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    mi00 = jnp.ones((npnts, nlayers + 1), jnp.complex128)

    sld = jnp.zeros(nlayers + 2, jnp.complex128)

    # addition of TINY is to ensure the correct branch cut
    # in the complex sqrt calculation of kn.
    sld = index_add(
        sld,
        index[1:],
        ((layers[1:, 1] - layers[0, 1]) + 1j * (jnp.abs(layers[1:, 2]) + TINY))
        * 1.0e-6,
    )
    kn = jnp.sqrt(flatq[:, jnp.newaxis] ** 2.0 / 4.0 - 4.0 * jnp.pi * sld)
    # reflectances for each layer
    # rj.shape = (npnts, nlayers + 1)
    damping = jnp.exp(-2.0 * kn[:, :-1] * kn[:, 1:] * layers[1:, 3] ** 2)
    rj = (kn[:, :-1] - kn[:, 1:]) / (kn[:, :-1] + kn[:, 1:]) * damping

    # characteristic matrices for each layer
    # miNN.shape = (npnts, nlayers + 1)
    if nlayers:
        mi00 = index_update(
            mi00,
            index[:, 1:],
            jnp.exp(kn[:, 1:-1] * 1j * jnp.fabs(layers[1:-1, 0])),
        )
    mi11 = 1.0 / mi00
    mi10 = rj * mi11
    mi01 = rj * mi00

    mi = jnp.zeros((npnts, nlayers + 1, 2, 2), jnp.complex128)

    mi = index_update(
        mi,
        index[:, :, 0, 0],
        mi00,
    )
    mi = index_update(
        mi,
        index[:, :, 0, 1],
        mi01,
    )
    mi = index_update(
        mi,
        index[:, :, 1, 1],
        mi11,
    )
    mi = index_update(
        mi,
        index[:, :, 1, 0],
        mi10,
    )

    sub = [jnp.squeeze(v) for v in jnp.hsplit(mi, nlayers + 1)]
    mrtot = reduce(jnp.matmul, sub[1:], sub[0])

    r = mrtot[:, 1, 0] / mrtot[:, 0, 0]
    reflectivity = r * jnp.conj(r) * scale
    reflectivity = index_add(reflectivity, ..., bkg)

    return jnp.real(jnp.reshape(reflectivity, qvals.shape))


# abeles_jax = jabeles
abeles_jax = jit(jabeles)
