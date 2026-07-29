"""
*Calculates the specular (Neutron or X-ray) reflectivity from a stratified
series of layers, using the C `abeles` kernel (src/refcalc.c) as an XLA FFI
custom call.

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

Notes
-----
`abeles_jax_ffi` is a drop-in replacement for `abeles_jax` (`_jax_reflect.py`):
same `(q, layers, scale=1.0, bkg=0.0)` signature, same numerical result. The
forward pass runs the hand-vectorised C kernel via jax's FFI mechanism; the
backward pass (`jax.grad` / `jax.value_and_grad`) is piggybacked onto
`jabeles`'s ordinary jax autodiff via a `jax.custom_vjp` rule, so no gradient
math is hand-derived here.

This only supports reverse-mode AD (grad/value_and_grad). It does *not*
currently support jax.jacfwd or forward-mode jvp -- jax.custom_vjp functions
raise a TypeError under forward-mode AD. jax.vmap also isn't supported --
jax.ffi.ffi_call raises a NotImplementedError unless a vmap_method is
specified, and that's deliberately left unset here: passing e.g.
vmap_method="broadcast_all" runs without error but silently returns wrong
results for this FFI target, since the packed coefP layout has no batch
dimension the C kernel understands. Use `abeles_jax` from `_jax_reflect.py`
for jacfwd/vmap.

Requires `jax_enable_x64`:

    from jax import config
    config.update("jax_enable_x64", True)
"""

import ctypes

import jax
import jax.numpy as jnp

from refnx.reflect._jax_reflect import jabeles

_TARGET_NAME = "abeles_jax_ffi"
_registered = False


def _register():
    global _registered
    if _registered:
        return

    try:
        from refnx.reflect import _abeles_jax_ffi as _abeles_jax_ffi_ext
    except ImportError as exc:
        raise ImportError(
            "The _abeles_jax_ffi extension module is not available. refnx must "
            "be built with jax importable in the build environment for "
            "this module to work (see refnx/reflect/meson.build)."
        ) from exc

    # _abeles_jax_ffi_ext.__file__ points at the actual built shared object --
    # the build directory in an editable install, the installed package
    # directory otherwise -- so it can be handed straight to ctypes.
    lib = ctypes.cdll.LoadLibrary(_abeles_jax_ffi_ext.__file__)
    jax.ffi.register_ffi_target(
        _TARGET_NAME, jax.ffi.pycapsule(lib.Abeles_JAX_FFI), platform="cpu"
    )
    _registered = True


def _pack_coefs(layers, scale, bkg):
    # Mirrors the coefP layout documented in src/refcalc.h and built in
    # src/_creflect.pyx: [nlayers, scale, fronting(re,im), backing(re,im),
    # bkg, backing_rough, then 4 values per layer: thick, re, im, rough].
    nlayers = layers.shape[0] - 2
    head = jnp.stack(
        [
            jnp.asarray(nlayers, dtype=jnp.float64),
            jnp.asarray(scale, dtype=jnp.float64),
            layers[0, 1],
            layers[0, 2],
            layers[-1, 1],
            layers[-1, 2],
            jnp.asarray(bkg, dtype=jnp.float64),
            layers[-1, 3],
        ]
    )
    if nlayers:
        body = layers[1:-1, :4].reshape(-1)
        return jnp.concatenate([head, body])
    return head


@jax.custom_vjp
def _abeles_jax_ffi_core(layers, scale, bkg, q):
    _register()
    coefP = _pack_coefs(layers, scale, bkg)
    out_type = jax.ShapeDtypeStruct(q.shape, jnp.float64)
    call = jax.ffi.ffi_call(_TARGET_NAME, out_type)
    return call(coefP, q)


def _abeles_jax_ffi_fwd(layers, scale, bkg, q):
    y = _abeles_jax_ffi_core(layers, scale, bkg, q)
    return y, (layers, scale, bkg, q)


def _abeles_jax_ffi_bwd(res, ct):
    layers, scale, bkg, q = res

    def f(l, s, b):
        return jabeles(q, l, s, b)

    _, vjp_fn = jax.vjp(f, layers, scale, bkg)
    d_layers, d_scale, d_bkg = vjp_fn(ct)
    return (d_layers, d_scale, d_bkg, None)


_abeles_jax_ffi_core.defvjp(_abeles_jax_ffi_fwd, _abeles_jax_ffi_bwd)


def abeles_jax_ffi(q, layers, scale=1.0, bkg=0.0, threads=1):
    """
    Calculate specular reflectivity using the C `abeles` kernel via jax FFI.

    Drop-in replacement for `abeles_jax` (same signature, same numerics).
    The forward pass is the fast, hand-vectorised C implementation; gradients
    (`jax.grad`, `jax.value_and_grad`) are computed by differentiating the
    pure-jax `jabeles` implementation instead, via a custom VJP -- see the
    module docstring for what is and isn't supported.

    Parameters
    ----------
    q : array-like
        Q values, Angstrom**-1.
    layers : jnp.ndarray
        Layer stack, shape (N, 4), as consumed by `jabeles`.
    scale : float or jnp.ndarray
    bkg : float or jnp.ndarray
    threads : int
        ignored

    Returns
    -------
    reflectivity : jnp.ndarray
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    flatq = q.ravel()
    scale = jnp.asarray(scale, dtype=jnp.float64)
    bkg = jnp.asarray(bkg, dtype=jnp.float64)

    r = _abeles_jax_ffi_core(layers, scale, bkg, flatq)
    return jnp.reshape(r, q.shape)
