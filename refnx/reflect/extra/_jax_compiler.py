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
import warnings

import numpy as np
import jax.numpy as jnp
import jax
from refnx.reflect import LipidLeaflet, LipidLeafletGuest
from refnx.reflect.extra._jax_util import (
    _SlabSpec,
    _PaddedSlabSpecs,
    _ConstNode,
    _ConstraintCompiler,
    _eval_node,
)
from refnx.reflect.extra._jax_lipid import (
    _lipid_leaflet_jax_slabs,
    _lipid_leaflet_guest_jax_slabs,
)

# monkey patch known Component classes
_jax_slabs_methods = {
    LipidLeaflet: _lipid_leaflet_jax_slabs,
    LipidLeafletGuest: _lipid_leaflet_guest_jax_slabs,
}
for klass, method in _jax_slabs_methods.items():
    klass._jax_slabs = method

# ---------------------------------------------------------------------------
# Compile a Structure's slab layout into JAX
# ---------------------------------------------------------------------------


def _compile_structure(
    structure, compiler: _ConstraintCompiler
) -> List[_SlabSpec]:
    """
    Walk a refnx Structure and return a list of ``_SlabSpec`` or
    ``_PaddedSlabSpecs`` objects — one entry per slab row (or per padded
    block for variable-slab components that opt in to ``_PaddedSlabSpecs``).

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
            # General extension point: may return List[_SlabSpec] or
            # _PaddedSlabSpecs for components with bounded-variable slab counts.
            result = component._jax_slabs(compiler)
            if isinstance(result, _PaddedSlabSpecs):
                specs.append(result)
            else:
                specs.extend(result)
        elif hasattr(component, "thick") and type(component) is Slab:
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
        else:
            raise ValueError(
                f"_jax_slabs is currently not implemented for {type(component)}"
            )
            # Unknown multi-slab component (e.g. Spline, LipidLeaflet):
            # bake current numeric values as constants.
            # AD will not flow through these slabs; document this clearly.
            # for row in component_slabs:
            #     # row has at least 5 columns: thick, real, imag, rough, vfsolv
            #     specs.append(
            #         _SlabSpec(
            #             _ConstNode(float(row[0])),
            #             _ConstNode(float(row[1])),
            #             _ConstNode(float(row[2])),
            #             _ConstNode(float(row[3])),
            #             _ConstNode(float(row[4])),
            #         )
            #     )

    return specs


def _compile_solvent(structure, compiler: _ConstraintCompiler):
    """
    Compile the solvent SLD of a ``Structure`` into a pair of IR nodes
    ``(real_node, imag_node)``.

    ``Structure.solvent`` can be:

    1. ``None`` — solvent is inferred from the last component's SLD.
       We read the last ``Slab``'s ``sld.real`` / ``sld.imag`` Parameters
       and compile them, so gradients flow if those are free.

    2. An ``SLD`` object set explicitly by the user — its ``real`` and
       ``imag`` attributes are ``Parameter`` objects that may be free.
       We compile them directly.

    When ``structure.reverse_structure=True``, the solvating medium is
    the *first* component (which becomes the backing after reversal), so
    we compile from ``structure.components[0]`` instead.
    """
    from refnx.analysis.parameter import is_parameter
    from refnx.reflect import SLD

    if structure.reverse_structure:
        # After reversal the first component becomes the backing/solvent.
        anchor = structure.components[0]
    else:
        anchor = structure.components[-1]

    # If structure.solvent is an explicit SLD object, compile its Parameters.
    if isinstance(structure.solvent, SLD):
        real_node = compiler.compile_parameter(structure.solvent.real)
        imag_node = compiler.compile_parameter(structure.solvent.imag)
        return real_node, imag_node

    # Otherwise derive from the anchor component's slab SLD Parameters.
    if hasattr(anchor, "sld"):
        real_node = compiler.compile_parameter(anchor.sld.real)
        imag_node = compiler.compile_parameter(anchor.sld.imag)
    else:
        # Fallback for exotic components: evaluate once and bake as constant.
        slabs = anchor.slabs(structure=structure)
        real_node = _ConstNode(float(slabs[0, 1]))
        imag_node = _ConstNode(float(slabs[0, 2]))

    return real_node, imag_node


def _make_params_to_slabs(
    slab_specs: List[_SlabSpec],
    solvent_real_node,
    solvent_imag_node,
    reverse_structure: bool = False,
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
        Compiled slab specifications from ``_compile_structure``, always in
        forward (fronting-to-backing) order.
    solvent_real_node, solvent_imag_node : IR node
        Compiled IR nodes for the real and imaginary SLD of the solvating
        medium (×10⁻⁶ Å⁻²).  These may be ``_ConstNode`` values (when the
        solvent SLD is fixed) or ``_FreeNode`` / expression nodes (when
        ``structure.solvent`` is an ``SLD`` object with fittable parameters).
        When ``reverse_structure=True`` these should represent the *fronting*
        medium SLD (i.e. the original first component), since after reversal
        the roles of fronting and backing are swapped.
    reverse_structure : bool
        When True, reverse the slab order and shift the roughness column to
        match ``Structure.slabs()`` behaviour.  Handled entirely at compile
        time — no runtime branching.
    """
    # Apply reverse_structure at compile time by reordering slab_specs.
    # Structure.slabs() does:
    #   slabs = slabs[::-1]
    #   slabs[1:, 3] = slabs[:-1, 3][::-1]   # shift roughnesses
    #   slabs[0, 3] = 0.0                     # fronting roughness is 0
    # We replicate this by reordering the _SlabSpec list and reassigning
    # the rough fields before the JAX function is built — zero extra cost
    # at runtime.
    if reverse_structure:
        specs = list(reversed(slab_specs))
        original_roughs = [s.rough for s in slab_specs]
        shifted_roughs = [_ConstNode(0.0)] + list(reversed(original_roughs))[
            :-1
        ]
        specs = [
            _SlabSpec(s.thick, s.real, s.imag, r, s.vfsolv)
            for s, r in zip(specs, shifted_roughs)
        ]
    else:
        specs = slab_specs

    def params_to_slabs(free: jnp.ndarray) -> jnp.ndarray:
        # Evaluate solvent SLD nodes — free if solvent has fittable parameters,
        # constant otherwise.
        solv_re = _eval_node(solvent_real_node, free)
        solv_im = _eval_node(solvent_imag_node, free)

        # Build (N, 5): thick, sld_re, sld_im, rough, vfsolv
        # Each entry in specs is either a _SlabSpec (one row) or a
        # _PaddedSlabSpecs (n_max rows with a dynamic active-count mask).
        rows = []
        for spec in specs:
            if isinstance(spec, _PaddedSlabSpecs):
                # Evaluate all n_max rows, then zero out inactive thicknesses.
                # jnp.arange < n_active produces a boolean mask; multiplying
                # thick by it makes inactive rows contribute nothing in Abeles.
                n_active = _eval_node(spec.n_active_node, free)
                mask = jnp.arange(spec.n_max) < n_active  # (n_max,) bool
                for i, s in enumerate(spec.specs):
                    thick = _eval_node(s.thick, free) * mask[i]
                    real = _eval_node(s.real, free)
                    imag = _eval_node(s.imag, free)
                    rough = _eval_node(s.rough, free)
                    vfsolv = _eval_node(s.vfsolv, free)
                    rows.append(jnp.stack([thick, real, imag, rough, vfsolv]))
            else:
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


def _make_generative(
    params_to_slabs: Callable,
    q: jnp.ndarray,
    q_err: Optional[jnp.ndarray],
    scale_node,
    bkg_node,
    quad_order: int = 17,
) -> Callable:
    """
    Returns a pure JAX function ``free -> R(q)`` (shape ``(N,)``).

    This is the forward model — the reflectivity curve predicted by the
    current parameter values — without any likelihood computation.  It is
    the shared inner kernel used by both ``_make_logl`` and exposed directly
    as ``CompiledObjective.generative``.

    Parameters
    ----------
    params_to_slabs : Callable
        Pure JAX function ``free -> (N, 4)`` layers array.
    q : (N,) jnp.ndarray
        Q values, frozen at compile time.
    q_err : (N,) jnp.ndarray or None
        Per-point absolute dQ values.  When not None, smearing is applied.
    scale_node, bkg_node : IR nodes
        Compiled scale and background.
    quad_order : int
        Gauss-Legendre order for smearing.
    """
    from refnx.reflect._jax_reflect import (
        jabeles,
        jax_smeared_kernel_pointwise,
    )

    use_smearing = q_err is not None

    def generative(free: jnp.ndarray) -> jnp.ndarray:
        layers = params_to_slabs(free)
        scale = _eval_node(scale_node, free)
        bkg = _eval_node(bkg_node, free)
        if use_smearing:
            return jax_smeared_kernel_pointwise(
                q, layers, q_err, quad_order=quad_order, scale=scale, bkg=bkg
            )
        else:
            return jabeles(q, layers, scale=scale, bkg=bkg)

    return generative


def _make_logl(
    generative: Callable,
    y: jnp.ndarray,
    y_err: Optional[jnp.ndarray],
    lnsigma_node,
    use_weights: bool,
) -> Callable:
    """
    Returns a pure JAX function ``free -> scalar log-likelihood``.

    Delegates the forward model to ``generative(free) -> R(q)``, which is
    built by ``_make_generative``.  This eliminates duplication: the same
    ``generative`` function is also exposed as
    ``CompiledObjective.generative``.

    Mirrors Objective.logl:

        var = y_err**2 + exp(2*lnsigma) * model**2   (if lnsigma present)
        logl = -0.5 * sum( (y - model)**2/var + log(2*pi*var) )
    """

    def logl_jax(free: jnp.ndarray) -> jnp.ndarray:
        model = generative(free)

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
            params_to_slabs : Callable
    params_to_slabs : Callable
        Pure JAX function mapping the free vector to the (N, 4) layers array
        consumed by ``jabeles`` (thick, mixed_sld_real, mixed_sld_imag, rough).
        Solvent volume-fraction averaging has already been applied.
        Useful for inspecting the SLD profile under AD.
    setp : Callable[[np.ndarray], None]
        Push values back into the stateful objective after use.
    n_free : int
        Number of free parameters.
    """

    model: Callable
    jacfwd: Callable
    x0: jnp.ndarray
    param_names: List[str]
    params_to_slabs: Callable
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
    solvent_real_node, solvent_imag_node = _compile_solvent(
        structure, compiler
    )
    params_to_slabs_fn = _make_params_to_slabs(
        slab_specs,
        solvent_real_node,
        solvent_imag_node,
        reverse_structure=structure.reverse_structure,
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
        params_to_slabs=params_to_slabs_fn,
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
    generative : Callable[[jnp.ndarray], jnp.ndarray]
        Pure JAX forward model ``free -> R(q)`` (shape ``(N,)``).
        JIT-compiled.  Evaluates the reflectivity curve at the Q values and
        resolution of the objective's dataset. Useful for posterior predictive
        checks and plotting the model curve at optimised parameters.
        For a ``GlobalObjective`` this is the concatenation of the
        per-objective generative functions, matching the order of
        ``GlobalObjective.objectives``.
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
    generative: Callable
    x0: jnp.ndarray
    param_names: List[str]
    setp: Callable
    n_free: int


def _compile_single(
    objective, compiler: _ConstraintCompiler, quad_order: int = 17
) -> tuple:
    """
    Compile one ``Objective`` into raw (un-JIT-ted) JAX functions.

    This is the shared workhorse called by both ``compile_objective`` and
    ``compile_global_objective``.  It builds the generative model first,
    then passes it to ``_make_logl`` so both share exactly the same forward
    model computation.

    Parameters
    ----------
    objective : refnx.analysis.Objective
    compiler : _ConstraintCompiler
        Already constructed with the correct ``free_index``.
    quad_order : int
        Gauss-Legendre quadrature order for resolution smearing.

    Returns
    -------
    logl_raw : Callable[[jnp.ndarray], jnp.ndarray]
        Pure JAX log-likelihood, not yet JIT-compiled.
    generative_raw : Callable[[jnp.ndarray], jnp.ndarray]
        Pure JAX forward model ``free -> R(q)``, not yet JIT-compiled.
    params_to_slabs_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Pure JAX function mapping the free vector to the (N, 4) layers array.
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

    solvent_real_node, solvent_imag_node = _compile_solvent(
        structure, compiler
    )
    params_to_slabs_fn = _make_params_to_slabs(
        slab_specs,
        solvent_real_node,
        solvent_imag_node,
        reverse_structure=structure.reverse_structure,
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

    q = jnp.array(objective.data.x, dtype=jnp.float64)
    y = jnp.array(objective.data.y, dtype=jnp.float64)
    y_err = (
        jnp.array(objective.data.y_err, dtype=jnp.float64)
        if objective.weighted
        else None
    )

    dqvals = None
    _quad_order = model.quad_order

    if model.dq_type == "pointwise" and objective.data.x_err is None:
        if model.dq.value > 0:
            dqvals = model.dq.value / 100.0 * objective.data.x
    if model.dq_type == "constant" and model.dq.value > 0:
        dqvals = model.dq.value / 100.0 * objective.data.x
    if model.dq_type == "pointwise" and objective.data.x_err is not None:
        dqvals = jnp.array(objective.data.x_err, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Build the generative model first — it is the shared forward model.
    # ------------------------------------------------------------------
    generative_raw = _make_generative(
        params_to_slabs_fn, q, dqvals, scale_node, bkg_node, _quad_order
    )

    # ------------------------------------------------------------------
    # Pass generative into _make_logl so the same computation is reused.
    # ------------------------------------------------------------------
    logl_raw = _make_logl(
        generative_raw,
        y,
        y_err,
        lnsigma_node,
        use_weights=objective.weighted,
    )

    # ------------------------------------------------------------------
    # model.logp() and logp_extra
    # ------------------------------------------------------------------
    has_logp_extra = objective.logp_extra is not None

    # Check once at compile time whether model.logp is the trivial default
    # (the base Model.logp which always returns 0) and logp_extra is absent.
    # If so, skip the callback entirely to avoid unnecessary setp overhead.
    from refnx.analysis.model import Model as _BaseModel

    _logp_is_trivial = (
        not has_logp_extra and type(model).logp is _BaseModel.logp
    )

    if _logp_is_trivial:
        return logl_raw, generative_raw, params_to_slabs_fn

    if has_logp_extra:
        warnings.warn(
            "An Objective specifies logp_extra. Derivatives"
            " aren't being tracked",
            category=RuntimeWarning,
        )

    # Build a compile-time index array that maps from the global free vector
    # (which may be longer than this objective's own varying parameters when
    # called from compile_global_objective) to the values this objective's
    # setp() expects.  For a single Objective the two are identical; for a
    # GlobalObjective child they are a strict subset.
    local_var_params = list(objective.varying_parameters())
    setp_indices = np.array(
        [compiler._free_index[id(p)] for p in local_var_params], dtype=np.intp
    )

    def _extra_potential_callback(free_np: np.ndarray) -> np.ndarray:
        objective.setp(free_np[setp_indices])
        extra = float(model.logp())
        if has_logp_extra:
            extra += float(objective.logp_extra(model, objective.data))
        return np.array(extra, dtype=np.float64)

    @jax.custom_jvp
    def _extra_potential(free: jnp.ndarray) -> jnp.ndarray:
        return jax.pure_callback(
            _extra_potential_callback,
            jax.ShapeDtypeStruct((), jnp.float64),
            free,
        )

    @_extra_potential.defjvp
    def _extra_potential_jvp(primals, tangents):
        (free,) = primals
        return _extra_potential(free), jnp.zeros((), dtype=jnp.float64)

    def logl_with_extra(free: jnp.ndarray) -> jnp.ndarray:
        return logl_raw(free) + _extra_potential(free)

    return logl_with_extra, generative_raw, params_to_slabs_fn


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
    logl_raw, generative_raw, params_to_slabs_fn = _compile_single(
        objective, compiler
    )

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
        generative=jax.jit(generative_raw),
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
    generative_fns: List[Callable] = []

    for obj, lam in zip(global_objective.objectives, global_objective.lambdas):
        logl_i, generative_i, _ = _compile_single(obj, compiler)
        weighted_logl_fns.append((float(lam), logl_i))
        generative_fns.append(generative_i)

    def generative_global(free: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([fn(free) for fn in generative_fns])

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
        params_to_slabs=None,
        generative=jax.jit(generative_global),
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
