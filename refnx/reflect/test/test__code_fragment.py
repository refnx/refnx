import os.path
import operator

import numpy as np
from numpy import array
from numpy.testing import assert_allclose

from refnx.analysis import Objective, Parameter, Interval, Transform
from refnx.analysis.parameter import build_constraint_from_tree, Constant
from refnx.reflect._code_fragment import code_fragment
from refnx.reflect import SLD, ReflectModel, Structure, Slab
from refnx.dataset import ReflectDataset
from refnx._lib import flatten


class TestCodeFragment:
    def setup_method(self):
        self.pth = os.path.dirname(os.path.abspath(__file__))

    def test_code_fragment(self):
        e361 = ReflectDataset(os.path.join(self.pth, "e361r.txt"))

        si = SLD(2.07, name="Si")
        sio2 = SLD(3.47, name="SiO2")
        d2o = SLD(6.36, name="D2O")
        polymer = SLD(1, name="polymer")

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

        objective = Objective(self.model361, e361, transform=Transform("logY"))
        objective2 = eval(repr(objective))
        assert_allclose(objective2.chisqr(), objective.chisqr())

        exec(repr(objective))
        exec(code_fragment(objective))

        # artificially link the two thicknesses together
        # check that we can reproduce the objective from the repr
        self.structure361[2].thick.constraint = self.structure361[1].thick
        fragment = code_fragment(objective)
        fragment = fragment + "\nobj = objective()\nresult = obj.chisqr()"
        d = {}
        # need to provide the globals dictionary to exec, so it can see imports
        # e.g. https://bit.ly/2RFOF7i (from stackoverflow)
        exec(fragment, globals(), d)
        assert_allclose(d["result"], objective.chisqr())
