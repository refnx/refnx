import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_,
    assert_allclose,
)
from scipy.stats import norm
from refnx.reflect import Erf, Exponential, Step, Linear, Tanh, Sinusoidal


class TestStructure:
    def setup_method(self):
        self.x = np.linspace(-5, 5, 1001)

    def test_erf(self):
        i = Erf()
        profile = i(self.x, scale=1.1, loc=-1.0)

        assert_equal(profile, norm.cdf(self.x, scale=1.1, loc=-1.0))

    def test_exp(self):
        i = Exponential()
        i(self.x, scale=1.1, loc=-1.0)

    def test_linear(self):
        i = Linear()
        i(self.x, scale=1.1, loc=-1.0)

    def test_step(self):
        i = Step()
        i(self.x, scale=1.1, loc=-1.0)

    def test_sin(self):
        i = Sinusoidal()
        i(self.x, scale=1.1, loc=-1.0)

    def test_Tanh(self):
        i = Tanh()
        i(self.x, scale=1.1, loc=-1.0)

    def test_repr(self):
        cls = [Erf, Exponential, Step, Linear, Tanh, Sinusoidal]

        for c in cls:
            o = c()
            p = eval(repr(o))
            assert isinstance(p, c)
