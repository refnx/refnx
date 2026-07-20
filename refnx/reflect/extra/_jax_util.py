from typing import Any, Callable, Dict, List, Optional, Tuple

import operator
from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp


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


# ---------------------------------------------------------------------------
# Analyse the Parameter graph and build the IR
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
        from refnx.analysis.parameter import (
            Parameter,
            BaseParameter,
            _BinaryOp,
            _UnaryOp,
        )

        if isinstance(p, (_BinaryOp, _UnaryOp)):
            return self.compile(p)

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
        left = _eval_node(node.left, free)
        right = _eval_node(node.right, free)
        return node.op(left, right)
    elif isinstance(node, _UnaryNode):
        o = _eval_node(node.operand, free)
        return node.op(o)
    elif isinstance(node, _CallableConstraintNode):
        args = [_eval_node(a, free) for a in node.arg_nodes]
        return node.fn(*args)
    else:
        raise TypeError(f"Unknown IR node type: {type(node)}")


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
# IR arithmetic helpers
# ---------------------------------------------------------------------------
# We build IR nodes using the same node classes as the compiler, but we never
# import them at module level to avoid a hard JAX dependency at import time.


def _div(a, b):
    """IR node for a / b."""
    return _BinaryNode(operator.truediv, a, b)


def _mul(a, b):
    """IR node for a * b."""
    return _BinaryNode(operator.mul, a, b)


def _sub(a, b):
    """IR node for a - b."""
    return _BinaryNode(operator.sub, a, b)


def _add(a, b):
    """IR node for a + b."""
    return _BinaryNode(operator.add, a, b)


def _const(v: float):
    return _ConstNode(float(v))
