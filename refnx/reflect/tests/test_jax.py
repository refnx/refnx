from importlib import resources
import pytest
import numpy as np
from numpy.testing import assert_allclose

import refnx
from refnx.analysis import Objective, Parameter
from refnx.dataset import Data1D
from refnx.reflect import ReflectModel, SLD
from refnx.reflect.extra import compile_objective, compile_model

try:
    import jax
    from jax import config

    config.update("jax_enable_x64", True)
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False


@pytest.mark.skipif(not HAVE_JAX, reason="Requires jax")
class TestJAX:
    def setup_method(self):
        self.pth = resources.files(refnx.reflect.tests)

        air = SLD(0, name="air")
        quartz = SLD(5, name="quartz")
        sio2 = SLD(4.2, name="SiO2")
        si = SLD(2.07, name="Si")

        s = self.structure = air | quartz(1500, 5.0) | sio2(10, 5) | si(0, 5.0)

        quartz.real.setp(vary=True, bounds=(0, 5.0))
        sio2.real.setp(vary=True, bounds=(0, 5.0))
        si.real.setp(vary=True, bounds=(0, 5.0))

        s[1].thick.setp(vary=True, bounds=(1400.0, 1500.0))
        s[1].rough.setp(vary=True, bounds=(2.0, 20.0))

        s[2].thick.setp(vary=True, bounds=(0.0, 50.0))
        s[2].rough.setp(vary=True, bounds=(2.0, 20.0))

        s[-1].rough.setp(vary=True, bounds=(2.0, 20.0))

        bkg = Parameter(1e-7, name="bkg", vary=True, bounds=(1e-20, 1))
        scale = Parameter(1.0, name="scale", vary=True, bounds=(0.9, 1.5))

        model = self.model = ReflectModel(s, bkg=bkg, scale=scale)

        data = np.loadtxt(self.pth / ".Quartz_data.txt", delimiter=",")
        data = data[:, 1:]
        data = Data1D(data.T, name="data")

        # q-resolution column is a standard deviation
        data.x_err *= 2.3542

        self.objective = Objective(model, data)

    def test_compile_objective(self):
        # obtain the negative log-likelihood (nll) from the compiled objective (quick_test)
        # by looking at the nll we're implicitly checking resolution smearing, nll calculation,
        # etc
        obj = compile_objective(self.objective)
        vg = obj.value_and_grad
        logl, grad = vg(np.array(self.objective.varying_parameters()))
        assert_allclose(-logl, self.objective.nll())
