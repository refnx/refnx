"""
Test co-refinement of datasets by fitting 3 neutron reflectivity datasets. The
overall construction of the models can be done in a few different ways.
"""
import os.path

import numpy as np
from numpy.testing import (
    assert_,
    assert_equal,
    assert_almost_equal,
    assert_allclose,
)

from refnx.analysis import CurveFitter, Objective, GlobalObjective, Transform
from refnx.dataset import ReflectDataset
from refnx.reflect import Slab, SLD, ReflectModel

SEED = 1


class TestGlobalFitting:
    def setup_method(self):
        self.pth = os.path.dirname(os.path.abspath(__file__))

        self.si = SLD(2.07, name="Si")
        self.sio2 = SLD(3.47, name="SiO2")
        self.d2o = SLD(6.36, name="d2o")
        self.h2o = SLD(-0.56, name="h2o")
        self.cm3 = SLD(3.5, name="cm3")
        self.polymer = SLD(2, name="polymer")

        self.sio2_l = self.sio2(40, 3)
        self.polymer_l = self.polymer(200, 3)

        self.structure = (
            self.si | self.sio2_l | self.polymer_l | self.d2o(0, 3)
        )

        fname = os.path.join(self.pth, "c_PLP0011859_q.txt")

        self.dataset = ReflectDataset(fname)
        self.model = ReflectModel(self.structure, bkg=2e-7)
        self.objective = Objective(
            self.model,
            self.dataset,
            use_weights=False,
            transform=Transform("logY"),
        )
        self.global_objective = GlobalObjective([self.objective])

    def test_residuals_length(self):
        # the residuals should be the same length as the data
        residuals = self.global_objective.residuals()
        assert_equal(residuals.size, len(self.dataset))

    def test_lambdas(self):
        assert np.size(self.global_objective.lambdas) == 1
        self.global_objective.lambdas[0] = 5.0
        assert_allclose(
            self.global_objective.logl(), self.objective.logl() * 5.0
        )

    def test_globalfitting(self):
        # smoke test for can the global fitting run?
        # also tests that global fitting gives same output as
        # normal fitting (for a single dataset)
        self.model.scale.setp(vary=True, bounds=(0.1, 2))
        self.model.bkg.setp(vary=True, bounds=(1e-10, 8e-6))
        self.structure[-1].rough.setp(vary=True, bounds=(0.2, 6))
        self.sio2_l.thick.setp(vary=True, bounds=(0.2, 80))
        self.polymer_l.thick.setp(bounds=(0.01, 400), vary=True)
        self.polymer_l.sld.real.setp(vary=True, bounds=(0.01, 4))

        self.objective.transform = Transform("logY")

        starting = np.array(self.objective.parameters)
        with np.errstate(invalid="raise"):
            g = CurveFitter(self.global_objective)
            res_g = g.fit()

            # need the same starting point
            self.objective.setp(starting)
            f = CurveFitter(self.objective)
            res_f = f.fit()

            # individual and global should give the same fit.
            assert_almost_equal(res_g.x, res_f.x)

    def test_multipledataset_corefinement(self):
        # test corefinement of three datasets
        data361 = ReflectDataset(os.path.join(self.pth, "e361r.txt"))
        data365 = ReflectDataset(os.path.join(self.pth, "e365r.txt"))
        data366 = ReflectDataset(os.path.join(self.pth, "e366r.txt"))

        si = SLD(2.07, name="Si")
        sio2 = SLD(3.47, name="SiO2")
        d2o = SLD(6.36, name="d2o")
        h2o = SLD(-0.56, name="h2o")
        cm3 = SLD(3.47, name="cm3")
        polymer = SLD(1, name="polymer")

        structure361 = si | sio2(10, 4) | polymer(200, 3) | d2o(0, 3)
        structure365 = si | structure361[1] | structure361[2] | cm3(0, 3)
        structure366 = si | structure361[1] | structure361[2] | h2o(0, 3)

        structure365[-1].rough = structure361[-1].rough
        structure366[-1].rough = structure361[-1].rough

        structure361[1].thick.setp(vary=True, bounds=(0, 20))
        structure361[2].thick.setp(
            value=200.0, bounds=(200.0, 250.0), vary=True
        )
        structure361[2].sld.real.setp(vary=True, bounds=(0, 2))
        structure361[2].vfsolv.setp(value=5.0, bounds=(0.0, 100.0), vary=True)

        model361 = ReflectModel(structure361, bkg=2e-5)
        model365 = ReflectModel(structure365, bkg=2e-5)
        model366 = ReflectModel(structure366, bkg=2e-5)

        model361.bkg.setp(vary=True, bounds=(1e-6, 5e-5))
        model365.bkg.setp(vary=True, bounds=(1e-6, 5e-5))
        model366.bkg.setp(vary=True, bounds=(1e-6, 5e-5))

        objective361 = Objective(model361, data361)
        objective365 = Objective(model365, data365)
        objective366 = Objective(model366, data366)

        global_objective = GlobalObjective(
            [objective361, objective365, objective366]
        )
        # are the right numbers of parameters varying?
        assert_equal(len(global_objective.varying_parameters()), 7)

        # can we set the parameters?
        global_objective.setp(np.array([1e-5, 10, 212, 1, 10, 1e-5, 1e-5]))

        f = CurveFitter(global_objective)
        f.fit()

        indiv_chisqr = np.sum(
            [objective.chisqr() for objective in global_objective.objectives]
        )

        # the overall chi2 should be sum of individual chi2
        global_chisqr = global_objective.chisqr()
        assert_almost_equal(global_chisqr, indiv_chisqr)

        # now check that the parameters were held in common correctly.
        slabs361 = structure361.slabs()
        slabs365 = structure365.slabs()
        slabs366 = structure366.slabs()

        assert_equal(slabs365[0:2, 0:5], slabs361[0:2, 0:5])
        assert_equal(slabs366[0:2, 0:5], slabs361[0:2, 0:5])
        assert_equal(slabs365[-1, 3], slabs361[-1, 3])
        assert_equal(slabs366[-1, 3], slabs361[-1, 3])

        # check that the residuals are the correct lengths
        res361 = objective361.residuals()
        res365 = objective365.residuals()
        res366 = objective366.residuals()
        res_global = global_objective.residuals()
        assert_allclose(res_global[0 : len(res361)], res361, rtol=1e-5)
        assert_allclose(
            res_global[len(res361) : len(res361) + len(res365)],
            res365,
            rtol=1e-5,
        )
        assert_allclose(
            res_global[len(res361) + len(res365) :], res366, rtol=1e-5
        )

        repr(global_objective)

        # test lagrangian multipliers
        global_objective.lambdas = np.array([1.1, 2.2, 3.3])
        assert_allclose(
            global_objective.logl(),
            1.1 * objective361.logl()
            + 2.2 * objective365.logl()
            + 3.3 * objective366.logl(),
        )
        global_objective.lambdas = np.ones(3)
