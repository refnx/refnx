"""
refnx JAX compiler
==================

Turns the stateful Parameter / Component graph into a *pure* JAX function that
maps a flat array of free-parameter values to a scalar log-likelihood.

The compiler is a one-shot analysis of the object graph performed **before**
the optimisation loop.  It emits:

  - a JAX-traceable ``params_to_slabs(free: jnp.ndarray) -> jnp.ndarray``
    that applies every constraint as a JAX expression, and
  - a JAX-traceable ``logl_jax(free: jnp.ndarray) -> jnp.ndarray``
    that calls jabeles and computes the Gaussian log-likelihood.

After compilation the stateful ``Objective`` is never touched again during
optimisation; only the pure functions are called.

Usage
-----
    from jax import config
    config.update("jax_enable_x64", True)

    from jax_compiler import compile_objective
    import jax

    logl_fn, grad_fn, meta = compile_objective(objective)

    # exact gradient — free of any Python side-effects
    x0 = meta["x0"]          # initial free-parameter values as jnp array
    val  = logl_fn(x0)
    grad = grad_fn(x0)

    # or together (cheaper):
    val, grad = jax.value_and_grad(logl_fn)(x0)

    # after optimisation, push values back into the stateful world:
    meta["setp"](best_x)     # calls objective.setp as usual
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import jax
from refnx.reflect import LipidLeaflet
from refnx.reflect.extra._jax_util import (
    _SlabSpec,
    _ConstNode,
    _ConstraintCompiler,
    _eval_node,
)
from refnx.reflect.extra._jax_lipid import _lipid_leaflet_jax_slabs

_jax_slabs_methods = {LipidLeaflet: _lipid_leaflet_jax_slabs}


# ---------------------------------------------------------------------------
# Compile a Structure's slab layout into JAX
# ---------------------------------------------------------------------------


def _compile_structure(
    structure, compiler: _ConstraintCompiler
) -> List[_SlabSpec]:
    """
    Walk a refnx Structure and return a list of _SlabSpec — one per slab
    row — using the current parameter values to locate each Parameter and
    compile its expression into the IR.

    This calls ``component.slabs()`` once to discover the *shape* of each
    component's contribution, then replaces each numeric value with an IR node
    that recomputes it from the free vector.

    The returned specs represent the *pre-mixing* five columns, i.e. the raw
    material SLD and vfsolv, not the solvent-averaged SLD.  Mixing is done
    inside ``_make_params_to_slabs`` so that gradients flow through it.
    """
    from refnx.reflect import Slab

    specs = []

    for component in structure.components:
        component_slabs = component.slabs(structure=structure)
        if component_slabs is None:
            continue

        n_rows = component_slabs.shape[0]

        if hasattr(component, "_jax_slabs"):
            # Extension point: component provides its own compiled slab specs.
            specs.extend(component._jax_slabs(compiler))
        elif hasattr(component, "thick") and isinstance(component, Slab):
            # Standard Slab interface (Slab, MagneticSlab, etc.)
            assert (
                n_rows == 1
            ), f"Component {component} has thick/sld/rough but {n_rows} slab rows."
            thick_node = compiler.compile_parameter(component.thick)
            real_node = compiler.compile_parameter(component.sld.real)
            imag_node = compiler.compile_parameter(component.sld.imag)
            rough_node = compiler.compile_parameter(component.rough)
            vfsolv_node = compiler.compile_parameter(component.vfsolv)
            specs.append(
                _SlabSpec(
                    thick_node, real_node, imag_node, rough_node, vfsolv_node
                )
            )
        elif isinstance(component, LipidLeaflet):
            LipidLeaflet._jax_slabs = _jax_slabs_methods[LipidLeaflet]
            specs.extend(component._jax_slabs(compiler))
        else:
            # Unknown multi-slab component (e.g. Spline, LipidLeaflet):
            # bake current numeric values as constants.
            # AD will not flow through these slabs; document this clearly.
            for row in component_slabs:
                # row has at least 5 columns: thick, real, imag, rough, vfsolv
                specs.append(
                    _SlabSpec(
                        _ConstNode(float(row[0])),
                        _ConstNode(float(row[1])),
                        _ConstNode(float(row[2])),
                        _ConstNode(float(row[3])),
                        _ConstNode(float(row[4])),
                    )
                )

    return specs


def _make_params_to_slabs(
    slab_specs: List[_SlabSpec], solvent_real: float, solvent_imag: float
):
    """
    Returns a pure JAX function  free -> (N, 4) layers array ready for
    ``jabeles``.

    Internally it first builds the full (N, 5) pre-mixing array
    ``[thick, sld_real, sld_imag, rough, vfsolv]``, then applies the
    volume-fraction solvent averaging that ``Structure.overall_sld`` performs
    in numpy:

        sld_mixed = sld_material * (1 - vfsolv) + sld_solvent * vfsolv

    for the interior slabs (indices 1 to -1), exactly mirroring
    ``Structure.slabs()``.  The fronting (index 0) and backing (index -1)
    slabs are never solvated.  The resulting (N, 4) slice is what jabeles
    consumes.

    Parameters
    ----------
    slab_specs : list of _SlabSpec
        Compiled slab specifications from ``_compile_structure``.
    solvent_real, solvent_imag : float
        Real and imaginary SLD of the solvating medium (×10⁻⁶ Å⁻²),
        captured as Python floats at compile time.
    """
    solv_re = float(solvent_real)
    solv_im = float(solvent_imag)

    def params_to_slabs(free: jnp.ndarray) -> jnp.ndarray:
        # Build (N, 5): thick, sld_re, sld_im, rough, vfsolv
        rows = []
        for spec in slab_specs:
            thick = _eval_node(spec.thick, free)
            real = _eval_node(spec.real, free)
            imag = _eval_node(spec.imag, free)
            rough = _eval_node(spec.rough, free)
            vfsolv = _eval_node(spec.vfsolv, free)
            rows.append(jnp.stack([thick, real, imag, rough, vfsolv]))
        slabs5 = jnp.stack(rows)  # (N, 5)

        # Apply solvent volume-fraction mixing on interior slabs,
        # replicating Structure.overall_sld / overall_sld().
        # Fronting (row 0) and backing (row -1) are left untouched.
        vf = slabs5[1:-1, 4:5]  # (N-2, 1)  vfsolv column
        re = slabs5[1:-1, 1] * (1.0 - vf[:, 0]) + solv_re * vf[:, 0]
        im = slabs5[1:-1, 2] * (1.0 - vf[:, 0]) + solv_im * vf[:, 0]

        # Rebuild the (N, 4) array jabeles expects: thick, mixed_re, mixed_im, rough
        interior = jnp.stack(
            [slabs5[1:-1, 0], re, im, slabs5[1:-1, 3]], axis=1
        )
        fronting = slabs5[0:1, :4]
        backing = slabs5[-1:, :4]
        return jnp.concatenate([fronting, interior, backing], axis=0)  # (N, 4)

    return params_to_slabs


# ---------------------------------------------------------------------------
# Compile the full log-likelihood
# ---------------------------------------------------------------------------


def _make_logl(
    params_to_slabs: Callable,
    q: jnp.ndarray,
    y: jnp.ndarray,
    y_err: Optional[jnp.ndarray],
    q_err: Optional[jnp.ndarray],
    scale_node,
    bkg_node,
    lnsigma_node,
    use_weights: bool,
    quad_order: int = 17,
):
    """
    Returns a pure JAX function  free -> scalar log-likelihood.

    ``params_to_slabs`` is expected to return a (N, 4) array (thick,
    sld_real_mixed, sld_imag_mixed, rough) with solvent mixing already
    applied — exactly what ``jabeles`` requires.

    When ``q_err`` is not None (i.e. the dataset carries per-point dQ/Q
    resolution), reflectivity is computed via
    ``jax_smeared_kernel_pointwise`` from ``refnx.reflect._jax_reflect``
    so that the gradient flows correctly through the smearing integral.
    When ``q_err`` is None the unsmeared ``jabeles`` is used directly.

    Mirrors Objective.logl:

        var = y_err**2 + exp(2*lnsigma) * model**2   (if lnsigma present)
        logl = -0.5 * sum( (y - model)**2/var + log(2*pi*var) )

    Parameters
    ----------
    q_err : (N,) jnp.ndarray or None
        Per-point dQ/Q values (FWHM) from ``Objective.data.x_err``.
        Pass None to skip resolution smearing.
    quad_order : int
        Gauss-Legendre quadrature order forwarded to
        ``jax_smeared_kernel_pointwise``.
    """
    from refnx.reflect._jax_reflect import (
        jabeles,
        jax_smeared_kernel_pointwise,
    )

    use_smearing = q_err is not None

    def logl_jax(free: jnp.ndarray) -> jnp.ndarray:
        layers = params_to_slabs(free)  # (N, 4) - solvent mixing already done
        scale = _eval_node(scale_node, free)
        bkg = _eval_node(bkg_node, free)

        if use_smearing:
            # jax_smeared_kernel_pointwise(qvals, w, dqvals, quad_order)
            # w is the layers array; dqvals are the absolute dQ FWHM values,
            # i.e. q_err (fractional dQ/Q) * q.
            model = jax_smeared_kernel_pointwise(
                q, layers, q_err, quad_order=quad_order, scale=scale, bkg=bkg
            )
        else:
            model = jabeles(q, layers, scale=scale, bkg=bkg)

        if use_weights and y_err is not None:
            base_var = y_err**2
        else:
            base_var = jnp.ones_like(y)

        if lnsigma_node is not None:
            lnsigma = _eval_node(lnsigma_node, free)
            var_y = base_var + jnp.exp(2.0 * lnsigma) * model**2
        else:
            var_y = base_var

        ll = (y - model) ** 2 / var_y
        if use_weights:
            ll = ll + jnp.log(2.0 * jnp.pi * var_y)

        return -0.5 * jnp.sum(ll)

    return logl_jax


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class CompiledModel:
    """
    The output of ``compile_model``.

    Mirrors the interface of ``CompiledObjective`` but exposes the
    forward model ``R(q; free)`` rather than the log-likelihood, so that
    gradients of the *reflectivity* with respect to free parameters are
    accessible directly.  Useful for sensitivity analysis, uncertainty
    propagation via linearisation, and visualising how R(q) changes with
    each parameter.

    Attributes
    ----------
    model : Callable[[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]
        Pure JAX function ``(free, q, q_err=None) -> R(q)`` (shape ``(N,)``).
        JIT-compiled.  Resolution smearing is applied when ``q_err`` is
        supplied and non-zero.
    jacfwd : Callable[[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]
        Forward-mode Jacobian ``(free, q, q_err=None) -> dR/d(free)``
        (shape ``(N, n_free)``).  JIT-compiled.  Differentiates w.r.t.
        ``free`` only; ``q`` and ``q_err`` are treated as fixed inputs.
    x0 : jnp.ndarray
        Initial free-parameter values.
    param_names : List[str]
        Names of the free parameters, in the same order as x0.
    setp : Callable[[np.ndarray], None]
        Push values back into the stateful objective after use.
    n_free : int
        Number of free parameters.
    """

    model: Callable
    jacfwd: Callable
    x0: jnp.ndarray
    param_names: List[str]
    setp: Callable
    n_free: int


def compile_model(reflect_model) -> CompiledModel:
    """
    Compile a ``ReflectModel`` into a pure JAX forward-model function.

    Returns a ``CompiledModel`` whose ``model(free, q, q_err=None)`` maps
    the free-parameter vector to the reflectivity curve R(q) for any
    supplied Q values, and whose ``jacfwd(free, q, q_err=None)`` gives the
    exact Jacobian dR_i/dp_j at those Q values.

    Accepting ``q`` and ``q_err`` at call time rather than compile time
    means a single ``CompiledModel`` can be evaluated and differentiated on
    any Q grid without recompilation — useful for simulation, sensitivity
    analysis on a fine grid, or fitting datasets with different Q ranges.

    Parameters
    ----------
    reflect_model : refnx.reflect.ReflectModel
        The model to compile.  Its ``structure``, ``scale``, and ``bkg``
        parameters are compiled into the JAX graph.  Any parameter that
        is currently set to vary will appear in the free vector.

    Returns
    -------
    compiled : CompiledModel

    Example
    -------
        cm = compile_model(model)

       # Jacobian dR/dp — shape (len(data.x), n_free):
        R = cm.model(cm.x0, data.x, data.x_err)

        # Jacobian on a finer synthetic grid, no smearing:
        q_fine = np.linspace(0.005, 0.3, 1000)
        J = cm.jacfwd(cm.x0, q_fine)

        # Jacobian on a finer synthetic grid, constant dq/q smearing:
        J = cm.jacfwd(cm.x0, q_fine, q_fine * 0.05)

    Notes
    -----
    Why jacfwd rather than jacrev?
    The Jacobian is (N_q, n_free) shaped. Forward-mode accumulates one column
    per free parameter (n_free passes); reverse-mode accumulates one row
    per Q point (N_q passes). For typical fits n_free is O(10–50) and
    N_q is O(100–1000), so forward mode is the right default. If you find
    yourself with far more parameters than Q points, jax.jacrev is a one-line swap.

    jacfwd vs jax.grad on a scalar reduction.
    For the log-likelihood jax.grad is ideal (one reverse pass). But model_fn
    returns a vector, so grad doesn't apply — jacfwd is the correct primitive here.
    """
    from refnx.analysis.parameter import is_parameter
    from refnx.reflect._jax_reflect import (
        jabeles,
        jax_smeared_kernel_pointwise,
    )

    var_params = list(reflect_model.parameters.varying_parameters())
    if not var_params:
        raise ValueError("ReflectModel has no varying parameters to compile.")

    free_index = {id(p): i for i, p in enumerate(var_params)}
    x0 = jnp.array([float(p.value) for p in var_params], dtype=jnp.float64)
    param_names = [p.name or f"p{i}" for i, p in enumerate(var_params)]
    compiler = _ConstraintCompiler(free_index)

    structure = reflect_model.structure
    slab_specs = _compile_structure(structure, compiler)
    solvent_complex = complex(structure.solvent.complex(structure.wavelength))
    params_to_slabs_fn = _make_params_to_slabs(
        slab_specs, solvent_complex.real, solvent_complex.imag
    )

    scale_node = (
        compiler.compile_parameter(reflect_model.scale)
        if is_parameter(reflect_model.scale)
        else _ConstNode(float(reflect_model.scale))
    )
    bkg_node = (
        compiler.compile_parameter(reflect_model.bkg)
        if is_parameter(reflect_model.bkg)
        else _ConstNode(float(reflect_model.bkg))
    )

    def model_fn(
        free: jnp.ndarray,
        q: jnp.ndarray,
        q_err: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        layers = params_to_slabs_fn(free)
        scale = _eval_node(scale_node, free)
        bkg = _eval_node(bkg_node, free)
        if q_err is not None:
            r = jax_smeared_kernel_pointwise(
                q,
                layers,
                q_err,
                scale=scale,
                bkg=bkg,
                quad_order=reflect_model.quad_order,
            )
            return r
        else:
            return jabeles(q, layers, scale=scale, bkg=bkg)

    def setp(x: np.ndarray) -> None:
        reflect_model.setp(np.asarray(x))

    # jacfwd differentiates w.r.t. the first argument (free) only;
    # q and q_err are treated as non-differentiated inputs.
    jacfwd_fn = jax.jacfwd(model_fn)

    return CompiledModel(
        model=jax.jit(model_fn),
        jacfwd=jax.jit(jacfwd_fn),
        x0=x0,
        param_names=param_names,
        setp=setp,
        n_free=len(var_params),
    )


@dataclass
class CompiledObjective:
    """
    The output of ``compile_objective``.

    Attributes
    ----------
    logl : Callable[[jnp.ndarray], jnp.ndarray]
        Pure JAX log-likelihood.  JIT-compiled.
    grad_logl : Callable[[jnp.ndarray], jnp.ndarray]
        Exact gradient of ``logl`` w.r.t. the free-parameter vector.
        JIT-compiled.
    value_and_grad : Callable
        ``jax.value_and_grad(logl)``, JIT-compiled.  Preferred for
        optimisers that need both simultaneously.
    params_to_slabs : Callable
        Pure JAX function mapping the free vector to the (N, 4) layers array
        consumed by ``jabeles`` (thick, mixed_sld_real, mixed_sld_imag, rough).
        Solvent volume-fraction averaging has already been applied.
        Useful for inspecting the SLD profile under AD.
    x0 : jnp.ndarray
        Initial free-parameter values extracted from the objective.
    param_names : List[str]
        Names of the free parameters, in the same order as x0.
    setp : Callable[[np.ndarray], None]
        Thin wrapper around ``objective.setp`` so you can push optimised
        values back into the stateful object graph after optimisation.
    n_free : int
        Number of free parameters.
    """

    logl: Callable
    grad_logl: Callable
    value_and_grad: Callable
    params_to_slabs: Callable
    x0: jnp.ndarray
    param_names: List[str]
    setp: Callable
    n_free: int


def _compile_single_logl(
    objective, compiler: _ConstraintCompiler, quad_order: int = 17
) -> Callable:
    """
    Compile one ``Objective`` into a raw (un-JIT-ted) JAX log-likelihood
    function using a pre-built ``_ConstraintCompiler``.

    This is the shared workhorse called by both ``compile_objective`` and
    ``compile_global_objective``.  Separating it out means the global case
    can build one ``free_index`` covering the full deduplicated parameter set,
    then compile each child objective against that same index so that shared
    parameters map to the same position in the free vector in every term.

    Parameters
    ----------
    objective : refnx.analysis.Objective
    compiler : _ConstraintCompiler
        Already constructed with the correct ``free_index``.
    quad_order : int
        Forwarded to ``_make_logl``.

    Returns
    -------
    logl_raw : Callable[[jnp.ndarray], jnp.ndarray]
        Pure JAX log-likelihood function, not yet JIT-compiled.
    params_to_slabs_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Pure JAX function mapping the free vector to the (N, 4) layers
        array for this objective's structure, not yet JIT-compiled.
    """
    from refnx.analysis.parameter import is_parameter

    model = objective.model
    if not hasattr(model, "structure"):
        raise TypeError(
            f"Objective '{objective.name}' has a model without a "
            "'structure' attribute.  Only ReflectModel-backed objectives "
            "are supported.  For other model types implement "
            "_jax_logl(compiler, data) on the model."
        )

    structure = model.structure
    slab_specs = _compile_structure(structure, compiler)

    solvent_complex = complex(structure.solvent.complex(structure.wavelength))
    params_to_slabs_fn = _make_params_to_slabs(
        slab_specs, solvent_complex.real, solvent_complex.imag
    )

    if hasattr(model, "scale") and is_parameter(model.scale):
        scale_node = compiler.compile_parameter(model.scale)
    elif hasattr(model, "scale"):
        scale_node = _ConstNode(float(model.scale))
    else:
        scale_node = _ConstNode(1.0)

    if hasattr(model, "bkg") and is_parameter(model.bkg):
        bkg_node = compiler.compile_parameter(model.bkg)
    elif hasattr(model, "bkg"):
        bkg_node = _ConstNode(float(model.bkg))
    else:
        bkg_node = _ConstNode(0.0)

    if objective.lnsigma is not None and is_parameter(objective.lnsigma):
        lnsigma_node = compiler.compile_parameter(objective.lnsigma)
    else:
        lnsigma_node = None

    # Compile auxiliary_params and alpha so that any free parameter among
    # them is registered in the IR and its constraint graph is walked.
    # This ensures gradients flow correctly through any constraint expression
    # that references an auxiliary parameter, and that setp_indices covers
    # the full set of parameters that objective.setp expects.
    # alpha only affects logpost (not logl) so its node is discarded here,
    # but compiling it ensures it is present in the free_index walk.
    for aux_p in objective.auxiliary_params:
        compiler.compile_parameter(aux_p)
    if objective.alpha is not None and is_parameter(objective.alpha):
        compiler.compile_parameter(objective.alpha)

    q = jnp.array(objective.data.x, dtype=jnp.float64)
    y = jnp.array(objective.data.y, dtype=jnp.float64)
    y_err = (
        jnp.array(objective.data.y_err, dtype=jnp.float64)
        if objective.weighted
        else None
    )

    # x_err holds per-point dQ/Q values (fractional FWHM resolution).
    # ReflectDataset stores None when no resolution information is present,
    # and a zero-filled array is treated as "no smearing" by ReflectModel.
    # We follow the same convention: only smear when x_err is non-None and
    # has at least one non-zero entry.
    dqvals = None
    quad_order = model.quad_order

    if model.dq_type == "pointwise" and objective.data.x_err is None:
        if model.dq.value > 0:
            dqvals = model.dq.value / 100.0 * objective.data.x
    if model.dq_type == "constant" and model.dq.value > 0:
        dqvals = model.dq.value / 100.0 * objective.data.x
    if model.dq_type == "pointwise" and objective.data.x_err is not None:
        dqvals = jnp.array(objective.data.x_err, dtype=jnp.float64)

    logl_raw = _make_logl(
        params_to_slabs_fn,
        q,
        y,
        y_err,
        dqvals,
        scale_node,
        bkg_node,
        lnsigma_node,
        use_weights=objective.weighted,
        quad_order=quad_order,
    )

    # ----------------------------------------------------------------------
    # model.logp() and logp_extra
    #
    # Objective.logl adds extra potential terms that are arbitrary Python
    # callables — they cannot be JAX-traced.  We use jax.pure_callback to
    # call them safely from within a JIT-compiled function: JAX treats the
    # callback as an opaque kernel, passes a concrete numpy array at runtime
    # (never an abstract tracer), and splices the returned scalar into the
    # JAX computation graph as a stop-gradient leaf.
    #
    # Gradients do NOT flow through these terms, which is correct: they
    # encode inequality constraints and regularisation expressed as Python
    # logic, not differentiable expressions.  Users who need differentiable
    # priors should express them as Parameter bounds or constraint
    # expressions in the Parameter graph instead.
    # ----------------------------------------------------------------------
    has_logp_extra = objective.logp_extra is not None

    # Build a compile-time mapping from each of this objective's local
    # varying parameters to their index in the global free vector.
    # We assign values individually by parameter identity so that ordering
    # differences between the global free vector and the local
    # varying_parameters() sequence cannot cause misassignment.
    local_var_params = list(objective.varying_parameters())
    local_param_indices = np.array(
        [compiler._free_index[id(p)] for p in local_var_params], dtype=np.intp
    )

    # Check once at compile time whether model.logp is the trivial default.
    from refnx.analysis.model import Model as _BaseModel

    _logp_is_trivial = (
        not has_logp_extra and type(model).logp is _BaseModel.logp
    )

    def _logl_callback(free_np: np.ndarray) -> np.ndarray:
        """
        Concrete-array callback: assign each local parameter value
        individually by identity, then evaluate the full logl.

        Assigning by identity rather than calling objective.setp with a
        sliced array avoids any sensitivity to ordering differences between
        the global free vector and the local varying_parameters() sequence.
        """
        for param, idx in zip(local_var_params, local_param_indices):
            param.value = free_np[idx]
        val = float(logl_raw(jnp.array(free_np, dtype=jnp.float64)))
        if not _logp_is_trivial:
            val += float(model.logp())
            if has_logp_extra:
                val += float(objective.logp_extra(model, objective.data))
        return np.array(val, dtype=np.float64)

    @jax.custom_jvp
    def logl_with_extra(free: jnp.ndarray) -> jnp.ndarray:
        return jax.pure_callback(
            _logl_callback,
            jax.ShapeDtypeStruct((), jnp.float64),
            free,
        )

    @logl_with_extra.defjvp
    def _logl_with_extra_jvp(primals, tangents):
        (free,) = primals
        (free_dot,) = tangents
        primal_out = logl_with_extra(free)
        _, tangent_out = jax.jvp(logl_raw, (free,), (free_dot,))
        return primal_out, tangent_out

    return logl_with_extra, params_to_slabs_fn


def compile_objective(objective) -> CompiledObjective:
    """
    Compile a refnx ``Objective`` into a pure JAX log-likelihood.

    The compilation analyses the Parameter graph *once*, at call time, and
    returns a ``CompiledObjective`` whose ``logl`` function is a pure,
    JIT-compiled JAX function with no Python side-effects.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        Must have a ``model`` that is a ``ReflectModel`` backed by a
        ``Structure`` composed of ``Slab``-like components.  More exotic
        components (e.g. ``Spline``) will have their slab values baked in as
        constants; AD will not flow through them unless they implement
        ``_jax_slabs(compiler)``.

    Returns
    -------
    compiled : CompiledObjective

    Notes
    -----
    **64-bit floats**: jabeles requires 64-bit.  Enable before calling::

        from jax import config
        config.update("jax_enable_x64", True)

    **Re-compilation**: if you change which parameters vary (or add/remove
    constraints) you must call ``compile_objective`` again.  The compiled
    function captures the graph topology at compile time.

    **lnsigma**: supported as a free or fixed parameter.

    **scale / background**: compiled from ``model.scale`` and ``model.bkg``
    Parameters if present; otherwise baked in as constants.

    **Callable constraints**: ``Parameter.set_constraint(fn, args=(...))``
    constraints are included, but gradients only flow through them if ``fn``
    itself uses JAX-compatible operations (jnp etc.).  numpy-based callables
    will act as stop-gradients.
    """
    # ------------------------------------------------------------------
    # 1.  Enumerate free parameters and build the index map
    # ------------------------------------------------------------------
    var_params = list(objective.varying_parameters())
    if len(var_params) == 0:
        raise ValueError("Objective has no varying parameters to compile.")

    free_index: Dict[int, int] = {id(p): i for i, p in enumerate(var_params)}
    x0 = jnp.array([float(p.value) for p in var_params], dtype=jnp.float64)
    param_names = [p.name or f"p{i}" for i, p in enumerate(var_params)]

    compiler = _ConstraintCompiler(free_index)

    # ------------------------------------------------------------------
    # 2–4.  Compile structure, scale/bkg/lnsigma, and freeze data arrays
    # ------------------------------------------------------------------
    logl_raw, params_to_slabs_fn = _compile_single_logl(objective, compiler)

    logl_jit = jax.jit(logl_raw)
    grad_jit = jax.jit(jax.grad(logl_raw))
    val_and_grad_jit = jax.jit(jax.value_and_grad(logl_raw))

    # ------------------------------------------------------------------
    # 6.  Convenience setp bridge
    # ------------------------------------------------------------------
    def setp(x: np.ndarray) -> None:
        """Push values from the free vector back into the stateful objective."""
        objective.setp(np.asarray(x))

    return CompiledObjective(
        logl=logl_jit,
        grad_logl=grad_jit,
        value_and_grad=val_and_grad_jit,
        params_to_slabs=jax.jit(params_to_slabs_fn),
        x0=x0,
        param_names=param_names,
        setp=setp,
        n_free=len(var_params),
    )


def compile_global_objective(global_objective) -> CompiledObjective:
    """
    Compile a refnx ``GlobalObjective`` into a pure JAX log-likelihood.

    The compiled log-likelihood is the weighted sum of the individual
    objective log-likelihoods, mirroring ``GlobalObjective.logl``:

        logl = sum( lambda_i * logl_i(free) )

    Because all child objectives are compiled against a **single shared
    free-parameter index** built from ``global_objective.varying_parameters()``,
    parameters that are shared between objectives (e.g. a polymer SLD
    refined simultaneously against multiple contrasts) appear exactly once
    in the free vector and their gradients are summed correctly by JAX's
    autodiff — no special handling is required.

    Parameters
    ----------
    global_objective : refnx.analysis.GlobalObjective

    Returns
    -------
    compiled : CompiledObjective
        The ``params_to_slabs`` field is not populated for a
        ``GlobalObjective`` (it is set to ``None``) because there is no
        single layer stack — use the individual compiled objectives if you
        need per-contrast SLD profiles.

    Notes
    -----
    **lambdas**: ``GlobalObjective.lambdas`` are plain floats and are baked
    in as constants at compile time.  If you change them you must
    recompile.

    **All other notes from** ``compile_objective`` **apply to each child
    objective individually** (64-bit floats, re-compilation on topology
    change, callable constraints, etc.).

    Example
    -------
        compiled = compile_global_objective(global_objective)
        val, grad = compiled.value_and_grad(compiled.x0)

        # push back into the stateful world after optimisation:
        compiled.setp(result.x)
    """
    # ------------------------------------------------------------------
    # 1.  Build one shared free-parameter index from the global varying set.
    #     GlobalObjective.varying_parameters() already returns the
    #     deduplicated union, in a stable order, so shared parameters
    #     appear exactly once.
    # ------------------------------------------------------------------
    var_params = list(global_objective.varying_parameters())
    if not var_params:
        raise ValueError(
            "GlobalObjective has no varying parameters to compile."
        )

    free_index: Dict[int, int] = {id(p): i for i, p in enumerate(var_params)}
    x0 = jnp.array([float(p.value) for p in var_params], dtype=jnp.float64)
    param_names = [p.name or f"p{i}" for i, p in enumerate(var_params)]

    compiler = _ConstraintCompiler(free_index)

    # ------------------------------------------------------------------
    # 2.  Compile each child objective against the shared compiler.
    #     The lambda weights are baked in as Python floats (constants in
    #     the XLA graph) because GlobalObjective.lambdas are plain floats.
    # ------------------------------------------------------------------
    weighted_logl_fns: List[tuple] = []  # (lambda_float, logl_raw_fn)

    for obj, lam in zip(global_objective.objectives, global_objective.lambdas):
        logl_i, _ = _compile_single_logl(obj, compiler)
        weighted_logl_fns.append((float(lam), logl_i))

    # ------------------------------------------------------------------
    # 3.  Combine into a single pure JAX function by summing the weighted
    #     log-likelihoods.  The sum is unrolled at Python level (not with
    #     jnp.sum over a stacked array) so that each term's XLA graph is
    #     independent — this lets XLA parallelise or fuse them freely.
    # ------------------------------------------------------------------
    def logl_global(free: jnp.ndarray) -> jnp.ndarray:
        total = jnp.zeros((), dtype=jnp.float64)
        for lam, logl_fn in weighted_logl_fns:
            total = total + lam * logl_fn(free)
        return total

    logl_jit = jax.jit(logl_global)
    grad_jit = jax.jit(jax.grad(logl_global))
    val_and_grad_jit = jax.jit(jax.value_and_grad(logl_global))

    # ------------------------------------------------------------------
    # 4.  setp bridge — GlobalObjective.setp handles the deduplication
    # ------------------------------------------------------------------
    def setp(x: np.ndarray) -> None:
        """Push values from the free vector back into the stateful world."""
        global_objective.setp(np.asarray(x))

    return CompiledObjective(
        logl=logl_jit,
        grad_logl=grad_jit,
        value_and_grad=val_and_grad_jit,
        params_to_slabs=None,  # no single layer stack for a global fit
        x0=x0,
        param_names=param_names,
        setp=setp,
        n_free=len(var_params),
    )


# ---------------------------------------------------------------------------
# Convenience: scipy-compatible (nll, grad) wrapper for L-BFGS-B etc.
# ---------------------------------------------------------------------------


def make_scipy_objective(compiled: CompiledObjective):
    """
    Returns ``(nll_fn, grad_fn)`` suitable for::

        scipy.optimize.minimize(nll_fn, x0, jac=grad_fn, method='L-BFGS-B')

    Both functions accept and return plain ``np.ndarray`` (float64).
    """
    val_grad = compiled.value_and_grad

    def nll(x: np.ndarray) -> float:
        v, _ = val_grad(jnp.array(x, dtype=jnp.float64))
        return float(-v)  # negative log-likelihood

    def grad_nll(x: np.ndarray) -> np.ndarray:
        _, g = val_grad(jnp.array(x, dtype=jnp.float64))
        return np.array(-g, dtype=np.float64)

    return nll, grad_nll
