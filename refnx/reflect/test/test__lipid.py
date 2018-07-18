from __future__ import division

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_allclose, assert_raises)

from refnx.reflect import (SLD, Structure, Spline, Slab, LipidLeaflet)
from refnx.reflect.structure import _profile_slicer
from refnx.analysis import Parameter


class TestLipidLeaflet(object):
    def setup_method(self):
        self.b_h = 6.01e-4
        self.V_h = 319.
        self.b_t = -2.92e-4
        self.V_t = 782.
        self.APM = 60.
        self.thick_h = 9.
        self.thick_t = 14.
        self.leaflet = LipidLeaflet(self.APM,
                                    self.b_h, self.V_h, self.thick_h,
                                    self.b_t, self.V_t, self.thick_t,
                                    2, 3)

        self.rho_h = self.b_h / self.V_h / 1e-6
        self.rho_t = self.b_t / self.V_t / 1e-6
        self.phi_solv_h = 1 - (self.V_h / (self.APM * self.thick_h))
        self.phi_solv_t = 1 - (self.V_t / (self.APM * self.thick_t))

    def test_slabs(self):
        # check that slab calculation from parameters is correct
        slabs = self.leaflet.slabs
        theoretical = np.array([[self.thick_h, self.rho_h, 0, 3,
                                 self.phi_solv_h],
                                [self.thick_t, self.rho_t, 0, 2,
                                 self.phi_solv_t]])
        assert_allclose(slabs, theoretical, rtol=1e-15)

        # check that we can flip the lipid leaflet
        self.leaflet.reverse_monolayer = True
        theoretical = np.flipud(theoretical)
        theoretical[:, 3] = theoretical[::-1, 3]
        assert_allclose(self.leaflet.slabs, theoretical, rtol=1e-15)

    def test_solvent_penetration(self):
        # check different types of solvation for heads/tails.
        self.leaflet.head_solvent = SLD(1.23)
        slabs = self.leaflet.slabs
        assert_allclose(slabs[0, 1], 1.23 * self.phi_solv_h +
                        (1 - self.phi_solv_h) * self.rho_h)
        assert_allclose(slabs[0, 4], 0)

        self.leaflet.head_solvent = None
        self.leaflet.tail_solvent = SLD(1.23)
        slabs = self.leaflet.slabs
        assert_allclose(slabs[1, 1], 1.23 * self.phi_solv_t +
                        (1 - self.phi_solv_t) * self.rho_t)
        assert_allclose(slabs[1, 4], 0)
