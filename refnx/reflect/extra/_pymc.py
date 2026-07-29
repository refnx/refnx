import warnings

import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import pytensor
from pytensor.link.jax.dispatch import jax_funcify

from refnx.analysis import GlobalObjective, process_chain
from refnx.analysis.objective import _to_pymc_distribution
from refnx.reflect.extra._jax_compiler import (
    compile_global_objective,
    compile_objective,
)


def to_pymc_model(objective, _dist=None):
    """
    Creates a pymc model from an Objective.

    Requires pytensor and pymc be installed. This is an experimental feature.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The Objective function to convert into a pymc model.

    _dist : {None, str}
        - None / 'normal' : uses ``pymc.NormalDist``, with observed y data on
            Objective.generative

        - 'potential' : uses ``pymc.Potential`` on Objective.logl

        - 'custom' : uses ``pymc.CustomDist`` on Objective.logl

        CustomDist/NormalDist can be used for model comparison/posterior
        predictive sampling/etc, Potential cannot.

        If any ReflectModel in the Objective has non-zero logp, or
        any Objective has a logp_extra function, then derivatives of
        those are not tracked. At that point it's advisable to just
        use 'potential', and not 'custom' or 'normal'.

        A ReflectModel might have a non-zero logp if there is a Component
        in its Structure that applies a probabilistic penalty.
        One such example might be LipidLeaflet if a volume fraction
        in a head/tail region exceeds 1. This is ok because the penalty
        is a non-differentiable (-np.inf or 0).
        However, if the penalty is used to steer the modelling process
        (ReflectModel.logp is finite and gets bigger/smaller depending
        on Parameters) then it's inadvisable to use anything but 'potential'.

        In benchmarking 'normal' seems to sample the fastest, but
        'potential' might be the most robust. This isn't a coincidence:
        None/'normal' evaluates the forward model and its gradient as two
        separate calls (``_GenerativeOp``/``_GenerativeVJPOp``), where the
        fast FFI-backed ``abeles_ffi`` kernel is used automatically.
        'potential'/'custom' fuse value and gradient into a single
        ``jax.value_and_grad`` call (``_LogLikeValueGradOp``), where
        ``abeles_ffi``'s gradient rule ends up re-doing the pure-JAX
        forward pass anyway -- so those two use the plain ``jabeles``
        kernel instead, chosen automatically based on ``_dist``.

    Returns
    -------
    model : pymc.Model

    Notes
    -----
    The varying parameters are renamed 'p0', 'p1', etc, as it's vital in pymc
    that all parameters have their own unique name.

    """
    import pymc as pm

    pars = objective.varying_parameters()
    wrapped_pars = []

    # 'potential'/'custom' (_LogLikeValueGradOp) fuse value+grad into a
    # single jax.value_and_grad call, where abeles_ffi's custom VJP has to
    # rebuild jabeles's own forward pass to get the gradient on top of its
    # own separately-paid-for C forward pass -- net slower than jabeles
    # directly there. 'normal'/None (_GenerativeOp/_GenerativeVJPOp) split
    # the forward and backward calls, where abeles_ffi's forward speedup is
    # real and uncompromised, so that path is left on compile_objective's
    # own default (abeles_ffi-preferring) reflect_fn.
    reflect_fn = None
    if _dist in ("potential", "custom"):
        from refnx.reflect._jax_reflect import jabeles

        reflect_fn = jabeles

    with warnings.catch_warnings():
        # raise an error if any Objective has logp_extra
        warnings.simplefilter("error", category=RuntimeWarning)

        if isinstance(objective, GlobalObjective):
            compiled_objective = compile_global_objective(
                objective, reflect_fn=reflect_fn
            )
            data = []
            y_err = []
            for _o in objective.objectives:
                data.append(_o.data.y)
                y_err.append(_o.data.y_err)
            data = np.concat(data, axis=0)
            y_err = np.concat(y_err, axis=0)
        else:
            compiled_objective = compile_objective(
                objective, reflect_fn=reflect_fn
            )
            data = objective.data.y
            y_err = objective.data.y_err

    with pm.Model() as basic_model:
        # Priors for unknown model parameters
        for i, par in enumerate(pars):
            name = "p%d" % i
            p = _to_pymc_distribution(name, par)
            wrapped_pars.append(p)

        theta = tuple(wrapped_pars)

        match _dist:
            case None | "normal":
                # Expected value of outcome
                gen_op = _GenerativeOp(compiled_objective)
                R_model = pm.Deterministic("R_model", gen_op(theta))
                pm.Normal(
                    "y_obs",
                    mu=R_model,
                    sigma=y_err,
                    observed=data,
                )
            case "potential":
                # Potential
                logl = _LogLikeValueGradOp(compiled_objective)
                pm.Potential("log-likelihood", logl(theta))

            case "custom":
                logl = _LogLikeValueGradOp(compiled_objective)

                def custom_dist_loglike(data, theta):
                    return logl(theta)

                pm.CustomDist(
                    "likelihood",
                    theta,
                    logp=custom_dist_loglike,
                    observed=data,
                )

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


class _GenerativeVJPOp(Op):

    def __init__(self, generative):
        self.generative = generative
        import jax

        # Compile the VJP once — called with concrete arrays at runtime.
        @jax.jit
        def _vjp_fn(free, cotangent):
            _, vjp = jax.vjp(generative, free)
            return vjp(cotangent)[0]

        self._vjp_fn = _vjp_fn

    def make_node(self, *inputs):
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        outputs = [pt.dvector()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        *free_scalars, cotangent = inputs
        free = np.array(free_scalars, dtype=np.float64)
        cotangent = np.asarray(cotangent, dtype=np.float64)
        outputs[0][0] = np.asarray(
            self._vjp_fn(free, cotangent), dtype=np.float64
        )


class _GenerativeOp(Op):
    """
    A pytensor ``Op`` that wraps ``CompiledObjective.generative``.

    Takes the tuple of free-parameter pytensor variables and returns
    a vector of predicted reflectivity values R(q) of shape ``(N_q,)``.
    """

    def __init__(self, compiled_objective):
        self.generative = compiled_objective.generative
        self._vjp_op = _GenerativeVJPOp(self.generative)

    def make_node(self, inputs):
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        outputs = [pt.dvector()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        free = np.array(inputs, dtype=np.float64)
        outputs[0][0] = np.asarray(self.generative(free), dtype=np.float64)

    def pullback(self, inputs, outputs, cotangents):
        # Express the VJP symbolically — _GenerativeVJPOp.perform is called
        # at runtime with concrete values, never at graph-construction time.
        grad = self._vjp_op(*inputs, cotangents[0])  # symbolic dvector
        # Return one gradient scalar per input
        return [grad[i] for i in range(len(inputs))]


def process_trace(objective, trace):
    """
    Process the trace produced by a pymc.sample run

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The Objective function that the Posterior was sampled on
    trace : trace
        The pymc sample trace

    Returns
    -------
    [(param, stderr, chain)] : list
        List of (param, stderr, chain) tuples.
        If `isinstance(objective.parameters, Parameters)` then `param` is a
        `Parameter` instance. `param.value`, `param.stderr` and
        `param.chain` will contain the median, stderr and chain samples,
        respectively. Otherwise `param` will be a float representing the
        median of the chain samples.
        `stderr` is the half width of the [15.87, 84.13] spread (similar to
        standard deviation) and `chain` is an array containing the MCMC
        samples for that parameter.

    Notes
    -----
    This function has the effect of setting the parameter stderr's.
    """
    varying_parameters = objective.varying_parameters()
    npars = len(varying_parameters)
    total_chain = [trace.posterior[f"p{i}"].to_numpy() for i in range(npars)]
    tc = np.r_[total_chain]
    output = process_chain(objective, np.swapaxes(tc, 0, 2))
    return output
