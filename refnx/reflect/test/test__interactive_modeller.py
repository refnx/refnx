import os.path
import os
import pickle

import numpy as np

from refnx.reflect import SLD, Slab, ReflectModel, Motofit
from refnx.dataset import ReflectDataset

from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_,
    assert_allclose,
)


class Test__InteractiveModeller:
    def setup_method(self):
        self.pth = os.path.dirname(os.path.abspath(__file__))

        sio2 = SLD(3.47, name="SiO2")
        air = SLD(0, name="air")
        si = SLD(2.07, name="Si")
        d2o = SLD(6.36, name="D2O")
        polymer = SLD(1, name="polymer")

        self.structure = air | sio2(100, 2) | si(0, 3)

        theoretical = np.loadtxt(os.path.join(self.pth, "theoretical.txt"))
        qvals, rvals = np.hsplit(theoretical, 2)
        self.qvals = qvals.flatten()
        self.rvals = rvals.flatten()

        # e361 is an older dataset, but well characterised
        self.structure361 = si | sio2(10, 4) | polymer(200, 3) | d2o(0, 3)
        self.model361 = ReflectModel(self.structure361, bkg=2e-5)

        self.model361.scale.vary = True
        self.model361.bkg.vary = True
        self.model361.scale.range(0.1, 2)
        self.model361.bkg.range(0, 5e-5)

        # d2o
        self.structure361[-1].sld.real.vary = True
        self.structure361[-1].sld.real.range(6, 6.36)

        self.structure361[1].thick.vary = True
        self.structure361[1].thick.range(5, 20)

        self.structure361[2].thick.vary = True
        self.structure361[2].thick.range(100, 220)

        self.structure361[2].sld.real.vary = True
        self.structure361[2].sld.real.range(0.2, 1.5)

        self.e361 = ReflectDataset(os.path.join(self.pth, "e361r.txt"))
        self.qvals361, self.rvals361, self.evals361 = (
            self.e361.x,
            self.e361.y,
            self.e361.y_err,
        )
        self.app = Motofit()
        self.app(self.e361, model=self.model361)

    def test_run_app(self):
        # figure out it some of the parameters are the same as you set them
        # with
        slab_views = self.app.model_view.structure_view.slab_views

        for slab_view, slab in zip(slab_views, self.structure361):
            assert_equal(slab_view.w_thick.value, slab.thick.value)
            assert_equal(slab_view.w_sld.value, slab.sld.real.value)
            assert_equal(slab_view.w_isld.value, slab.sld.imag.value)
            assert_equal(slab_view.w_rough.value, slab.rough.value)

            assert_equal(slab_view.c_thick.value, slab.thick.vary)
            assert_equal(slab_view.c_sld.value, slab.sld.real.vary)
            assert_equal(slab_view.c_isld.value, slab.sld.imag.vary)
            assert_equal(slab_view.c_rough.value, slab.rough.vary)

    def test_fit_runs(self):
        parameters = self.app.objective.parameters
        assert_(self.app.dataset is not None)
        assert_(self.app.model.structure[1].thick.vary is True)

        var_params = len(parameters.varying_parameters())
        assert_equal(var_params, 6)

        # sometimes the output buffer gets detached.
        try:
            self.app.do_fit(None)
        except ValueError:
            pass
