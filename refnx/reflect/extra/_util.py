import operator
from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp


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
