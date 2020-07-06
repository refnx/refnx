# log-likelihoods for pymc3.
# requires theano

import numpy as np
from scipy.optimize._numdiff import approx_derivative
import theano.tensor as tt


class _LogLikeWithGrad(tt.Op):
    # Theano op for calculating a log-likelihood

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        # add inputs as class attributes
        self.likelihood = loglike

        # initialise the gradient Op (below)
        self.logpgrad = _LogLikeGrad(self.likelihood)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters

        return [g[0] * self.logpgrad(theta)]


class _LogLikeGrad(tt.Op):
    # Theano op for calculating the gradient of a log-likelihood
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike):
        # add inputs as class attributes
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # define version of likelihood function to pass to derivative function
        def logl(values):
            return self.likelihood(values)

        # calculate gradients
        grads = approx_derivative(logl, theta, method="2-point")

        outputs[0][0] = grads
