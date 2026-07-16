import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op

from refnx.analysis import GlobalObjective
from refnx.analysis.objective import _to_pymc_distribution
from refnx.reflect.extra._jax_compiler import (
    compile_global_objective,
    compile_objective,
)


def _pymc_model(objective):
    """
    Creates a pymc model from an Objective.

    Requires aesara and pymc be installed. This is an experimental feature.

    Parameters
    ----------
    objective: refnx.analysis.Objective

    Returns
    -------
    model: pymc.Model

    Notes
    -----
    The varying parameters are renamed 'p0', 'p1', etc, as it's vital in pymc
    that all parameters have their own unique name.

    """
    import pymc as pm
    import pytensor.tensor as pt

    basic_model = pm.Model()

    pars = objective.varying_parameters()
    wrapped_pars = []

    if isinstance(objective, GlobalObjective):
        compiled_objective = compile_global_objective(objective)
    else:
        compiled_objective = compile_objective(objective)

    with basic_model:
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
        logl = _LogLikeValueGradOp(compiled_objective)
        pm.Potential("log-likelihood", logl(*theta))

    return basic_model


class _LogLikeValueGradOp(Op):
    default_output = 0

    def __init__(self, compiled_objective):
        self.compiled_objective = compiled_objective
        self.value_and_grad = compiled_objective.value_and_grad

    def make_node(self, *inputs):
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        # We now have one output for the function value, and one output for each gradient
        outputs = [pt.dscalar()] + [inp.type() for inp in inputs]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        print(inputs)
        result, grad_results = self.value_and_grad(np.asarray(inputs))
        outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)
        for i, grad_result in enumerate(grad_results, start=1):
            outputs[i][0] = np.asarray(
                grad_result, dtype=node.outputs[i].dtype
            )

    def grad(self, inputs, output_gradients):
        # The `Op` computes its own gradients, so we call it again.
        value = self(*inputs)
        # We hid the gradient outputs by setting `default_update=0`, but we
        # can retrieve them anytime by accessing the `Apply` node via `value.owner`
        gradients = value.owner.outputs[1:]

        # Make sure the user is not trying to take the gradient with respect to
        # the gradient outputs! That would require computing the second order
        # gradients
        assert all(
            isinstance(g.type, pytensor.gradient.DisconnectedType)
            for g in output_gradients[1:]
        )

        return [output_gradients[0] * grad for grad in gradients]


# class _LogLikeWithGrad(pt.Op):
#     # Theano op for calculating a log-likelihood
#
#     itypes = [pt.dvector]  # expects a vector of parameter values when called
#     otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)
#
#     def __init__(self, loglike):
#         # add inputs as class attributes
#         self.likelihood = loglike
#
#         # initialise the gradient Op (below)
#         self.logpgrad = _LogLikeGrad(self.likelihood)
#
#     def perform(self, node, inputs, outputs):
#         # the method that is used when calling the Op
#         (theta,) = inputs  # this will contain my variables
#
#         # call the log-likelihood function
#         logl = self.likelihood(theta)
#
#         outputs[0][0] = np.array(logl)  # output the log-likelihood
#
#     def grad(self, inputs, g):
#         # the method that calculates the gradients - it actually returns the
#         # vector-Jacobian product - g[0] is a vector of parameter values
#         (theta,) = inputs  # our parameters
#
#         return [g[0] * self.logpgrad(theta)]
#
#
# class _LogLikeGrad(pt.Op):
#     # Theano op for calculating the gradient of a log-likelihood
#     itypes = [pt.dvector]
#     otypes = [pt.dvector]
#
#     def __init__(self, loglike):
#         # add inputs as class attributes
#         self.likelihood = loglike
#
#     def perform(self, node, inputs, outputs):
#         (theta,) = inputs
#
#         # define version of likelihood function to pass to derivative function
#         def logl(values):
#             return self.likelihood(values)
#
#         # calculate gradients
#         grads = approx_derivative(logl, theta, method="2-point")
#
#         outputs[0][0] = grads
