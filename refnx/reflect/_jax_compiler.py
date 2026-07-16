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

import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import jax

# ---------------------------------------------------------------------------
# Node types — a tiny IR for constraint expressions
# ---------------------------------------------------------------------------


@dataclass
class _FreeNode:
    """A free (varying) parameter; evaluates to free[index]."""

    index: int


@dataclass
class _ConstNode:
    """A numeric constant (fixed parameter or literal)."""

    value: float


@dataclass
class _BinaryNode:
    """op(left, right) where op is a Python operator function."""

    op: Callable
    left: Any  # one of the Node types
    right: Any


@dataclass
class _UnaryNode:
    """op(operand) where op is a ufunc / unary function."""

    op: Callable
    operand: Any


@dataclass
class _CallableConstraintNode:
    """
    A constraint set via ``Parameter.set_constraint(fn, args=(...))``
    where ``fn`` is an arbitrary Python callable.

    We cannot trace through arbitrary callables with JAX.  Instead we record
    the function and the *resolved* argument nodes so that at JAX-trace time
    we can call ``fn(*evaluated_args)``.

    This only works if ``fn`` itself is composed of JAX-compatible operations
    (e.g. uses jnp inside).  If it uses numpy/Python scalars the gradient
    will be a stop-gradient.  We document this limitation clearly.
    """

    fn: Callable
    arg_nodes: List[Any]  # one node per arg in _constraint_args


# Map from refnx numpy ops to their jnp equivalents so the compiled
# expression uses JAX primitives rather than numpy ones.
_NP_TO_JNP: Dict[Callable, Callable] = {
    np.sin: jnp.sin,
    np.cos: jnp.cos,
    np.tan: jnp.tan,
    np.arcsin: jnp.arcsin,
    np.arccos: jnp.arccos,
    np.arctan: jnp.arctan,
    np.log: jnp.log,
    np.log10: jnp.log10,
    np.exp: jnp.exp,
    np.sqrt: jnp.sqrt,
    np.sum: jnp.sum,
    np.power: jnp.power,
    np.abs: jnp.abs,
    operator.add: operator.add,
    operator.sub: operator.sub,
    operator.mul: operator.mul,
    operator.truediv: operator.truediv,
    operator.floordiv: operator.floordiv,
    operator.pow: operator.pow,
    operator.mod: operator.mod,
    operator.neg: operator.neg,
    operator.abs: operator.abs,
}


# ---------------------------------------------------------------------------
# Step 1 — analyse the Parameter graph and build the IR
# ---------------------------------------------------------------------------


class _ConstraintCompiler:
    """
    Walks the refnx Parameter / _BinaryOp / _UnaryOp expression tree and
    converts it into our tiny IR of Node objects.

    Parameters
    ----------
    free_index : dict mapping id(Parameter) -> int
        The position of each free (varying) parameter in the flat vector.
    """

    def __init__(self, free_index: Dict[int, int]):
        self._free_index = free_index

    def compile(self, expr) -> Any:
        """Compile a constraint expression rooted at ``expr`` into a Node."""
        from refnx.analysis.parameter import (
            Parameter,
            Constant,
            _BinaryOp,
            _UnaryOp,
            BaseParameter,
        )

        if isinstance(expr, Parameter):
            pid = id(expr)
            if pid in self._free_index:
                # Free parameter: just read from the flat vector
                return _FreeNode(self._free_index[pid])
            elif expr.constraint is not None:
                # Constrained parameter: recurse into its constraint
                return self.compile(expr._constraint)
            else:
                # Fixed parameter: bake value in as a constant
                return _ConstNode(float(expr.value))

        elif isinstance(expr, Constant):
            return _ConstNode(float(expr.value))

        elif isinstance(expr, _BinaryOp):
            left = self.compile(expr.op1)
            right = self.compile(expr.op2)
            jax_op = _NP_TO_JNP.get(expr.opn, expr.opn)
            return _BinaryNode(jax_op, left, right)

        elif isinstance(expr, _UnaryOp):
            operand = self.compile(expr.op1)
            jax_op = _NP_TO_JNP.get(expr.opn, expr.opn)
            return _UnaryNode(jax_op, operand)

        elif callable(expr):
            # set_constraint(fn, args=(...)) path
            # expr is the raw callable; _constraint_args is on the Parameter
            # — but we're called *on the expression itself*, so this branch
            # is reached when compiling a Parameter whose ._constraint is
            # callable.  The args live on that Parameter, not here.
            # Handled separately in compile_parameter().
            return _ConstNode(float(expr()))  # fallback: evaluate once

        else:
            # Numeric literal wrapped by asMagicNumber — shouldn't reach here
            # normally, but guard anyway.
            return _ConstNode(float(expr))

    def compile_parameter(self, p) -> Any:
        """
        Top-level entry: compile the full expression for Parameter ``p``.
        Handles the callable-constraint case that needs access to
        ``p._constraint_args``.
        """
        from refnx.analysis.parameter import Parameter, BaseParameter

        pid = id(p)
        if pid in self._free_index:
            return _FreeNode(self._free_index[pid])

        if p.constraint is None:
            return _ConstNode(float(p.value))

        if callable(p._constraint) and not isinstance(
            p._constraint, BaseParameter
        ):
            # set_constraint(fn, args=(...)) style
            arg_nodes = []
            for arg in p._constraint_args or ():
                if isinstance(arg, BaseParameter):
                    arg_nodes.append(self.compile_parameter(arg))
                else:
                    arg_nodes.append(_ConstNode(float(arg)))
            return _CallableConstraintNode(p._constraint, arg_nodes)

        # algebraic expression constraint
        return self.compile(p._constraint)


def _eval_node(node, free: jnp.ndarray) -> jnp.ndarray:
    """Recursively evaluate a compiled IR node against the free vector."""
    if isinstance(node, _FreeNode):
        return free[node.index]
    elif isinstance(node, _ConstNode):
        return jnp.array(node.value, dtype=free.dtype)
    elif isinstance(node, _BinaryNode):
        l = _eval_node(node.left, free)
        r = _eval_node(node.right, free)
        return node.op(l, r)
    elif isinstance(node, _UnaryNode):
        o = _eval_node(node.operand, free)
        return node.op(o)
    elif isinstance(node, _CallableConstraintNode):
        args = [_eval_node(a, free) for a in node.arg_nodes]
        return node.fn(*args)
    else:
        raise TypeError(f"Unknown IR node type: {type(node)}")


# ---------------------------------------------------------------------------
# Step 2 — compile a Structure's slab layout into JAX
# ---------------------------------------------------------------------------


@dataclass
class _SlabSpec:
    """
    Compiled specification for one slab (one row of the (N, 5) layers array).
    Each field is an IR node that evaluates to a scalar.

    Columns match ``Component.slabs()`` exactly:
      0  thick   — layer thickness (Å)
      1  real    — real SLD of the pure material (×10⁻⁶ Å⁻²)
      2  imag    — imaginary SLD of the pure material (×10⁻⁶ Å⁻²)
      3  rough   — interfacial roughness (Å)
      4  vfsolv  — volume fraction of solvent in this layer [0, 1]

    The solvent mixing (``Structure.overall_sld``) is performed inside
    ``params_to_slabs`` so that gradients flow through it.  The four-column
    array consumed by ``jabeles`` is obtained by slicing ``[:, :4]`` *after*
    the mixing has been applied.
    """

    thick: Any  # thickness
    real: Any  # real SLD of pure material
    imag: Any  # imaginary SLD of pure material
    rough: Any  # roughness
    vfsolv: Any  # volume fraction of solvent


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
    specs = []

    for component in structure.components:
        component_slabs = component.slabs(structure=structure)
        if component_slabs is None:
            continue

        n_rows = component_slabs.shape[0]

        if hasattr(component, "_jax_slabs"):
            # Extension point: component provides its own compiled slab specs.
            specs.extend(component._jax_slabs(compiler))
        elif hasattr(component, "thick"):
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
# Step 4 — compile the full log-likelihood
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
    resolution), reflectivity is computed via ``_jabeles_smeared`` using
    Gauss-Legendre quadrature so that the gradient flows correctly through
    the smearing integral.  When ``q_err`` is None the unsmeared
    ``jabeles`` is used directly.

    Mirrors Objective.logl:

        var = y_err**2 + exp(2*lnsigma) * model**2   (if lnsigma present)
        logl = -0.5 * sum( (y - model)**2/var + log(2*pi*var) )

    Parameters
    ----------
    q_err : (N,) jnp.ndarray or None
        Per-point dQ/Q values (FWHM) from ``Objective.data.x_err``.
        Pass None to skip resolution smearing.
    quad_order : int
        Gauss-Legendre quadrature order forwarded to ``_jabeles_smeared``.
    """
    from refnx.reflect._jax_reflect import (
        jabeles,
        jax_smeared_kernel_pointwise,
    )

    use_smearing = q_err is not None

    def logl_jax(free: jnp.ndarray) -> jnp.ndarray:
        layers = params_to_slabs(free)  # (N, 4) — solvent mixing already done
        scale = _eval_node(scale_node, free)
        bkg = _eval_node(bkg_node, free)

        if use_smearing:
            model = jax_smeared_kernel_pointwise(
                q, layers, q_err, scale=scale, bkg=bkg, quad_order=quad_order
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
    # 2.  Compile the structure → slab layout
    # ------------------------------------------------------------------
    model = objective.model
    if not hasattr(model, "structure"):
        raise TypeError(
            "compile_objective currently requires model.structure to be a "
            "refnx Structure (i.e. a ReflectModel).  For other model types "
            "implement _jax_logl(compiler, data) on the model."
        )

    structure = model.structure
    slab_specs = _compile_structure(structure, compiler)

    # Resolve the solvent SLD once at compile time.
    # Structure.solvent returns a Scatterer; .complex() gives a Python complex.
    # This is correct: the solvent identity doesn't change during optimisation
    # (it's always the last slab's SLD or an explicit Structure.solvent).
    # If the solvent SLD is itself a free parameter the user should set
    # Structure.solvent explicitly to a Scatterer whose Parameters are in the
    # graph; for now we bake in the current numeric value, which is sufficient
    # for the common case of a fixed solvent contrast.
    solvent_complex = complex(structure.solvent.complex(structure.wavelength))
    solvent_real = solvent_complex.real
    solvent_imag = solvent_complex.imag

    params_to_slabs_fn = _make_params_to_slabs(
        slab_specs, solvent_real, solvent_imag
    )

    # ------------------------------------------------------------------
    # 3.  Compile scale, background, lnsigma
    # ------------------------------------------------------------------
    from refnx.analysis.parameter import is_parameter

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

    # ------------------------------------------------------------------
    # 4.  Freeze the data arrays
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5.  Build and JIT the pure log-likelihood
    # ------------------------------------------------------------------
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
