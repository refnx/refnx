import os

# Set these FIRST before any other imports
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative

try:
    import jax
    import jax.numpy as jnp
    from jax import config

    config.update("jax_enable_x64", True)
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False

if HAVE_JAX:
    try:
        import refnx.reflect._abeles_jax_ffi  # noqa: F401  (build-time probe)

        HAVE_ABELES_FFI = True
    except ImportError:
        HAVE_ABELES_FFI = False
else:
    HAVE_ABELES_FFI = False


pytestmark = pytest.mark.skipif(
    not HAVE_ABELES_FFI,
    reason="Requires jax and refnx built with the _abeles_jax_ffi extension",
)


# a handful of layer stacks, from zero layers up to several, to exercise the
# nlayers=0 edge case and the general recursion. Imaginary SLD is kept
# strictly positive everywhere: both the C kernel and jabeles apply
# fabs()/jnp.abs() to it, and jnp.abs has a real kink at exactly 0 (its
# subgradient there is 1.0, not the 0 a symmetric finite difference sees) --
# landing test data exactly on that kink would make gradient-checking
# ill-posed regardless of implementation, not a sign of a bug.
LAYER_STACKS = {
    "0-layer": np.array(
        [
            [0, 0, 0, 0],
            [0, 2.07, 1e-4, 3],
        ]
    ),
    "1-layer": np.array(
        [
            [0, 0, 0, 0],
            [200, 3.5, 0.0001, 3],
            [0, 2.07, 1e-4, 3],
        ]
    ),
    "3-layer": np.array(
        [
            [0, 0, 0, 0],
            [200, 3.5, 0.0001, 3],
            [50, 2.0, 0.0002, 4],
            [30, 1.5, 5e-5, 2],
            [0, 2.07, 1e-4, 3],
        ]
    ),
}


@pytest.fixture
def q():
    return np.linspace(0.005, 0.3, 250)


class TestAbelesFFI:
    @pytest.mark.parametrize("name", LAYER_STACKS)
    def test_forward_matches_creflect(self, q, name):
        # the C kernel invoked directly through jax's FFI should reproduce
        # the same C kernel invoked through the existing Cython bindings,
        # to float64 noise.
        from refnx.reflect._creflect import abeles as c_abeles
        from refnx.reflect._abeles_jax_ffi_wrapper import abeles_jax_ffi

        layers = jnp.asarray(LAYER_STACKS[name], dtype=jnp.float64)
        r_c = c_abeles(q, np.array(LAYER_STACKS[name]), scale=1.3, bkg=2e-7)
        r_ffi = abeles_jax_ffi(q, layers, scale=1.3, bkg=2e-7)
        assert_allclose(np.asarray(r_ffi), r_c, rtol=1e-10)

    @pytest.mark.parametrize("name", LAYER_STACKS)
    def test_forward_matches_jabeles(self, q, name):
        # cross-check against the pure-JAX reimplementation as well, since
        # that's what abeles_jax_ffi's own gradient rule is piggybacked on.
        from refnx.reflect._jax_reflect import abeles_jax
        from refnx.reflect._abeles_jax_ffi_wrapper import abeles_jax_ffi

        layers = jnp.asarray(LAYER_STACKS[name], dtype=jnp.float64)
        r_jax = abeles_jax(q, layers, scale=1.3, bkg=2e-7)
        r_ffi = abeles_jax_ffi(q, layers, scale=1.3, bkg=2e-7)
        assert_allclose(np.asarray(r_ffi), np.asarray(r_jax), rtol=1e-10)

    def test_gradient_matches_finite_difference(self, q):
        # Independent check: compare jax.grad(abeles_jax_ffi) against a
        # numerical finite-difference gradient, rather than against
        # jabeles's own autodiff (which abeles_jax_ffi's gradient rule is
        # piggybacked on, so that comparison alone would be somewhat
        # circular -- see test_gradient_matches_jabeles for that one too).
        from refnx.reflect._abeles_jax_ffi_wrapper import abeles_jax_ffi

        layers0 = LAYER_STACKS["3-layer"]
        # flatten (layers, scale, bkg) into one vector for approx_derivative
        x0 = np.concatenate([layers0.ravel(), [1.3, 2e-7]])

        def loss_np(x):
            layers = x[:-2].reshape(layers0.shape)
            scale, bkg = x[-2], x[-1]
            r = abeles_jax_ffi(q, jnp.asarray(layers), scale, bkg)
            return float(jnp.sum(r))

        def loss_jax(x):
            layers = jnp.reshape(x[:-2], layers0.shape)
            scale, bkg = x[-2], x[-1]
            return jnp.sum(abeles_jax_ffi(q, layers, scale, bkg))

        grad_ad = np.asarray(jax.grad(loss_jax)(jnp.asarray(x0)))
        grad_fd = approx_derivative(loss_np, x0, method="3-point")

        assert_allclose(grad_ad, grad_fd, rtol=1e-4, atol=1e-6)

    def test_gradient_matches_jabeles(self, q):
        # abeles_jax_ffi's gradient rule is explicitly piggybacked on jabeles's
        # autodiff (see _abeles_jax_ffi_wrapper.py) -- confirm the two stay in sync.
        from refnx.reflect._jax_reflect import abeles_jax
        from refnx.reflect._abeles_jax_ffi_wrapper import abeles_jax_ffi

        layers = jnp.asarray(LAYER_STACKS["3-layer"], dtype=jnp.float64)
        scale = jnp.float64(1.3)
        bkg = jnp.float64(2e-7)

        def loss_ffi(layers, scale, bkg):
            return jnp.sum(abeles_jax_ffi(q, layers, scale, bkg))

        def loss_jax(layers, scale, bkg):
            return jnp.sum(abeles_jax(q, layers, scale, bkg))

        v_ffi, g_ffi = jax.value_and_grad(loss_ffi, argnums=(0, 1, 2))(
            layers, scale, bkg
        )
        v_jax, g_jax = jax.value_and_grad(loss_jax, argnums=(0, 1, 2))(
            layers, scale, bkg
        )

        assert_allclose(v_ffi, v_jax, rtol=1e-10)
        for a, b in zip(g_ffi, g_jax):
            assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-8)

    def test_jacfwd_not_implemented(self, q):
        # forward-mode AD is not implemented (no jvp rule) -- this should
        # fail loudly rather than silently give a wrong answer.
        from refnx.reflect._abeles_jax_ffi_wrapper import abeles_jax_ffi

        layers = jnp.asarray(LAYER_STACKS["1-layer"], dtype=jnp.float64)
        with pytest.raises(NotImplementedError):
            jax.jacfwd(lambda l: abeles_jax_ffi(q, l, 1.0, 0.0))(layers)

    def test_vmap_not_implemented(self, q):
        # batching is not implemented (no batch/batch_dim_rule) -- same
        # reasoning as test_jacfwd_not_implemented.
        from refnx.reflect._abeles_jax_ffi_wrapper import abeles_jax_ffi

        layers1 = jnp.asarray(LAYER_STACKS["1-layer"], dtype=jnp.float64)
        layers2 = layers1.at[1, 0].set(
            210.0
        )  # same shape, different thickness
        layers_batch = jnp.stack([layers1, layers2])
        with pytest.raises(NotImplementedError):
            jax.vmap(lambda l: abeles_jax_ffi(q, l, 1.0, 0.0))(layers_batch)
