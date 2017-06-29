import unittest
import pickle

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_allclose, assert_raises)

from refnx.reflect import (SLD, Structure)
from refnx.analysis import Parameter


class TestStructure(unittest.TestCase):

    def setUp(self):
        self.air = SLD(0, name='air')
        self.sio2 = SLD(3.47, name='sio2')
        self.d2o = SLD(6.36, name='d2o')
        self.h2o = SLD(-0.56, name='h2o')
        self.s = self.air | self.sio2(100, 5) | self.d2o(0, 4)

    def test_structure_construction(self):
        # structures are constructed by or-ing slabs
        # test that the slab representation is correct
        assert_equal(self.s.slabs, np.array([[0, 0, 0, 0, 0],
                                             [100, 3.47, 0, 5, 0],
                                             [0, 6.36, 0, 4, 0]]))

        # slabs have solvent penetration
        self.s[1] = SLD(3.47 + 1j, name='sio2')(100, 5)
        self.s[1].solvent.value = 90.
        sld = 6.36 * 0.9 + 0.1 * 3.47
        sldi = 1 * 0.1
        assert_equal(self.s.slabs, np.array([[0, 0, 0, 0, 0],
                                             [100, sld, sldi, 5, 90],
                                             [0, 6.36, 0, 4, 0]]))

    def test_pickle(self):
        # need to be able to pickle and unpickle structure
        pkl = pickle.dumps(self.s)
        unpkl = pickle.loads(pkl)
        assert_(isinstance(unpkl, Structure))
        for param in unpkl.parameters.flattened():
            assert_(isinstance(param, Parameter))

    def test_sld_profile(self):
        # check that it runs
        z, sld_profile = self.s.sld_profile()

        z, sld_profile = self.s.sld_profile(np.linspace(-100, 100, 100))
        assert_equal(min(z), -100)
        assert_equal(max(z), 100)

    def test_reflectivity(self):
        q = np.geomspace(0.005, 0.3, 200)
        self.s.reflectivity(q)
