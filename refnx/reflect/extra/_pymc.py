import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import pytensor
from pytensor.link.jax.dispatch import jax_funcify

from refnx.analysis import GlobalObjective
from refnx.analysis.objective import _to_pymc_distribution
from refnx.reflect.extra._jax_compiler import (
    compile_global_objective,
    compile_objective,
)


def to_pymc_model(objective, _customdist=False):
    """
    Creates a pymc model from an Objective.

    Requires aesara and pymc be installed. This is an experimental feature.

    Parameters
    ----------
    objective: refnx.analysis.Objective

    _customdist: bool
        `True`:  uses ``pm.CustomDist``.
        `False`:  uses ``pm.Potential``.
        CustomDist can be used for model comparison, Potential cannot.

    Returns
    -------
    model: pymc.Model

    Notes
    -----
    The varying parameters are renamed 'p0', 'p1', etc, as it's vital in pymc
    that all parameters have their own unique name.

    """
    import pymc as pm

    pars = objective.varying_parameters()
    wrapped_pars = []

    if isinstance(objective, GlobalObjective):
        compiled_objective = compile_global_objective(objective)
        data = []
        for _o in objective.objectives:
            data.append(_o.data.y)
        data = np.concat(data, axis=0)
    else:
        compiled_objective = compile_objective(objective)
        data = objective.data.y

    with pm.Model() as basic_model:
        # Priors for unknown model parameters
        for i, par in enumerate(pars):
            name = "p%d" % i
            p = _to_pymc_distribution(name, par)
            wrapped_pars.append(p)

        # # Expected value of outcome
        # try:
        #     # Likelihood (sampling distribution) of observations
        #     pm.Normal(
        #         "y_obs",
        #         mu=objective.generative,
        #         sigma=objective.data.y_err,
        #         observed=objective.data.y,
        #     )
        # except Exception:
        #     # Falling back, theano autodiff won't work on function object
        # theta = pt.as_tensor_variable(wrapped_pars)
        theta = tuple(wrapped_pars)

        if _customdist:
            logl = _LogLikeValueGradOp(compiled_objective)

            def custom_dist_loglike(data, theta):
                return logl(theta)

            pm.CustomDist(
                "likelihood", theta, logp=custom_dist_loglike, observed=data
            )
        else:
            # Potential
            logl = _LogLikeValueGradOp(compiled_objective)
            pm.Potential("log-likelihood", logl(theta))

    return basic_model


class _LogLikeValueGradOp(Op):
    default_output = 0

    def __init__(self, compiled_objective):
        self.compiled_objective = compiled_objective
        self.value_and_grad = compiled_objective.value_and_grad

    def make_node(self, inputs):
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        # We now have one output for the function value, and one output for each gradient
        outputs = [pt.dscalar()] + [inp.type() for inp in inputs]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        result, grad_results = self.value_and_grad(np.asarray(inputs))
        outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)
        for i, grad_result in enumerate(grad_results, start=1):
            outputs[i][0] = np.asarray(
                grad_result, dtype=node.outputs[i].dtype
            )

    # def grad(self, inputs, output_gradients):
    #     # The `Op` computes its own gradients, so we call it again.
    #     value = self(inputs)
    #     # We hid the gradient outputs by setting `default_update=0`, but we
    #     # can retrieve them anytime by accessing the `Apply` node via `value.owner`
    #     gradients = value.owner.outputs[1:]
    #
    #     # Make sure the user is not trying to take the gradient with respect to
    #     # the gradient outputs! That would require computing the second order
    #     # gradients
    #     assert all(
    #         isinstance(g.type, pytensor.gradient.DisconnectedType)
    #         for g in output_gradients[1:]
    #     )
    #
    #     return [output_gradients[0] * grad for grad in gradients]

    def pullback(self, inputs, outputs, cotangents):
        # The Op computes its own gradients, so we call it again to
        # get at the hidden gradient outputs.
        value = self(inputs)
        gradients = value.owner.outputs[1:]

        # We don't support differentiating w.r.t. the gradient outputs
        # themselves (that would require second-order derivatives), so
        # their incoming cotangents must always be disconnected.
        assert all(
            isinstance(c.type, pytensor.gradient.DisconnectedType)
            for c in cotangents[1:]
        )

        return [cotangents[0] * grad for grad in gradients]


#
# @jax_funcify.register(_LogLikeValueGradOp)
# def jax_funcify_LogLikeValueGradOp(op, node=None, **kwargs):
#     import jax.numpy as jnp
#
#     value_and_grad = op.value_and_grad
#     n_params = len(node.inputs)
#
#     def perform(*inputs):
#         # stack along the last axis so any leading batch dims are preserved
#         theta = jnp.stack(inputs, axis=-1)  # shape (..., n_params)
#
#         if theta.ndim == 1:
#             # unbatched: single evaluation
#             value, grads = value_and_grad(theta)
#         else:
#             # batched (e.g. vmapped over chains by numpyro/blackjax)
#             batch_shape = theta.shape[:-1]
#             flat_theta = theta.reshape((-1, n_params))
#             value, grads = jax.vmap(value_and_grad)(flat_theta)
#             value = value.reshape(batch_shape)
#             grads = grads.reshape(batch_shape + (n_params,))
#
#         grad_outs = [grads[..., i] for i in range(n_params)]
#         return (jnp.asarray(value),) + tuple(grad_outs)
#
#     return perform


@jax_funcify.register(_LogLikeValueGradOp)
def jax_funcify_LogLikeValueGradOp(op, node=None, **kwargs):
    import jax
    import jax.numpy as jnp

    value_and_grad = op.value_and_grad
    n_params = len(node.inputs)

    # --- unbatched core: theta has shape (n_params,) ---
    @jax.custom_vjp
    def value_fn(theta):
        value, _ = value_and_grad(theta)
        return value

    def value_fwd(theta):
        value, grads = value_and_grad(theta)
        return value, grads  # residual = analytic grad, reused in bwd

    def value_bwd(grads, cotangent):
        return (cotangent * grads,)  # chain rule only, no re-differentiation

    value_fn.defvjp(value_fwd, value_bwd)

    # single evaluation gives both value and grad for the unbatched case
    value_and_grad_fn = jax.value_and_grad(value_fn)

    def perform(*inputs):
        theta = jnp.stack(
            inputs, axis=-1
        )  # (..., n_params); may carry a batch axis

        if theta.ndim == 1:
            value, grads = value_and_grad_fn(theta)
        else:
            # vmapped over leading batch dims (e.g. parallel chains under
            # numpyro/blackjax); vmap composes cleanly with custom_vjp
            batch_shape = theta.shape[:-1]
            flat_theta = theta.reshape((-1, n_params))
            value, grads = jax.vmap(value_and_grad_fn)(flat_theta)
            value = value.reshape(batch_shape)
            grads = grads.reshape(batch_shape + (n_params,))

        grad_outs = [grads[..., i] for i in range(n_params)]
        return (value,) + tuple(grad_outs)

    return perform
