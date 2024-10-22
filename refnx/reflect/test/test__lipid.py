import numpy as np
from pathlib import Path
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_allclose,
)
from scipy.optimize._constraints import PreparedConstraint
import refnx

# the analysis module contains the curvefitting engine
from refnx.analysis import CurveFitter, Objective

from refnx.reflect import (
    SLD,
    ReflectModel,
    Structure,
    Spline,
    Slab,
    LipidLeaflet,
    LipidLeafletGuest,
)
from refnx.reflect.structure import _profile_slicer
from refnx.analysis import Parameter, Interval

# the ReflectDataset object will contain the data
from refnx.dataset import ReflectDataset


class TestLipidLeaflet:
    def setup_method(self):
        self.b_h = 6.01e-4
        self.V_h = 319.0
        self.b_t = -2.92e-4
        self.V_t = 782.0
        self.APM = 60.0
        self.thick_h = 9.0
        self.thick_t = 14.0
        self.leaflet = LipidLeaflet(
            self.APM,
            self.b_h,
            self.V_h,
            self.thick_h,
            self.b_t,
            self.V_t,
            self.thick_t,
            2,
            3,
        )

        self.rho_h = self.b_h / self.V_h / 1e-6
        self.rho_t = self.b_t / self.V_t / 1e-6
        self.phi_solv_h = 1 - (self.V_h / (self.APM * self.thick_h))
        self.phi_solv_t = 1 - (self.V_t / (self.APM * self.thick_t))

    def test_slabs(self):
        # check that slab calculation from parameters is correct
        slabs = self.leaflet.slabs()
        theoretical = np.array(
            [
                [self.thick_h, self.rho_h, 0, 3, self.phi_solv_h],
                [self.thick_t, self.rho_t, 0, 2, self.phi_solv_t],
            ]
        )
        assert_allclose(slabs, theoretical, rtol=1e-15)

        # check that we can flip the lipid leaflet
        self.leaflet.reverse_monolayer = True
        theoretical = np.flipud(theoretical)
        theoretical[:, 3] = theoretical[::-1, 3]
        assert_allclose(self.leaflet.slabs(), theoretical, rtol=1e-15)

    def test_solvent_penetration(self):
        # check different types of solvation for heads/tails.
        self.leaflet.head_solvent = SLD(1.23)
        slabs = self.leaflet.slabs()
        assert_allclose(
            slabs[0, 1],
            1.23 * self.phi_solv_h + (1 - self.phi_solv_h) * self.rho_h,
        )
        assert_allclose(slabs[0, 4], 0)

        self.leaflet.head_solvent = None
        self.leaflet.tail_solvent = SLD(1.23)
        slabs = self.leaflet.slabs()
        assert_allclose(
            slabs[1, 1],
            1.23 * self.phi_solv_t + (1 - self.phi_solv_t) * self.rho_t,
        )
        assert_allclose(slabs[1, 4], 0)

    def test_initialisation_with_SLD(self):
        # we should be able to initialise with SLD objects
        heads = SLD(6.01e-4 + 0j)
        tails = SLD(-2.92e-4 + 0j)
        new_leaflet = LipidLeaflet(
            self.APM,
            heads,
            self.V_h,
            self.thick_h,
            tails,
            self.V_t,
            self.thick_t,
            2,
            3,
        )
        slabs = self.leaflet.slabs()
        new_slabs = new_leaflet.slabs()
        assert_allclose(new_slabs, slabs)

    def test_repr(self):
        # test that we can reconstruct the object from a repr
        s = repr(self.leaflet)
        q = eval(s)
        assert_equal(q.slabs(), self.leaflet.slabs())

    def test_lipidleafletguest(self):
        phi_guest = Parameter(0.1)
        sld_guest = SLD(7.6)
        leaflet = LipidLeafletGuest(
            self.APM,
            self.b_h,
            self.V_h,
            self.thick_h,
            self.b_t,
            self.V_t,
            self.thick_t,
            2,
            3,
            phi_guest,
            sld_guest,
        )
        leaflet.thickness_tails.value = 17.0
        assert isinstance(leaflet, LipidLeafletGuest)
        assert leaflet.phi_guest is phi_guest
        assert leaflet.sld_guest is sld_guest
        phi_t = leaflet.volfrac_t
        assert_equal(leaflet.volfrac_guest, (1 - phi_t) * phi_guest.value)


def test_lipid_leaflet_example():
    pth = Path(refnx.__file__).parent / "analysis" / "test"

    data_d2o = ReflectDataset(pth / "c_PLP0016596.dat")
    data_d2o.name = "d2o"

    si = SLD(2.07 + 0j)
    sio2 = SLD(3.47 + 0j)

    # the following represent the solvent contrasts used in the experiment
    d2o = SLD(6.36 + 0j)

    # We want the `real` attribute parameter to vary in the analysis, and we want to apply
    # uniform bounds. The `setp` method of a Parameter is a way of changing many aspects of
    # Parameter behaviour at once.
    d2o.real.setp(vary=True, bounds=(6.1, 6.36))
    d2o.real.name = "d2o SLD"

    # Parameter for the area per molecule each DMPC molecule occupies at the surface. We
    # use the same area per molecule for the inner and outer leaflets.
    apm = Parameter(56, "area per molecule", vary=True, bounds=(52, 65))

    # the sum of scattering lengths for the lipid head and tail in Angstrom.
    b_heads = Parameter(6.01e-4, "b_heads")
    b_tails = Parameter(-2.92e-4, "b_tails")

    # the volume occupied by the head and tail groups in cubic Angstrom.
    v_heads = Parameter(319, "v_heads")
    v_tails = Parameter(782, "v_tails")

    # the head and tail group thicknesses.
    inner_head_thickness = Parameter(
        9, "inner_head_thickness", vary=True, bounds=(4, 11)
    )
    outer_head_thickness = Parameter(
        9, "outer_head_thickness", vary=True, bounds=(4, 11)
    )
    tail_thickness = Parameter(
        14, "tail_thickness", vary=True, bounds=(10, 17)
    )

    # finally construct a `LipidLeaflet` object for the inner and outer leaflets.
    # Note that here the inner and outer leaflets use the same area per molecule,
    # same tail thickness, etc, but this is not necessary if the inner and outer
    # leaflets are different.
    inner_leaflet = LipidLeaflet(
        apm,
        b_heads,
        v_heads,
        inner_head_thickness,
        b_tails,
        v_tails,
        tail_thickness,
        3,
        3,
    )

    # we reverse the monolayer for the outer leaflet because the tail groups face upwards
    outer_leaflet = LipidLeaflet(
        apm,
        b_heads,
        v_heads,
        outer_head_thickness,
        b_tails,
        v_tails,
        tail_thickness,
        3,
        0,
        reverse_monolayer=True,
    )

    # Slab constructed from SLD object.
    sio2_slab = sio2(15, 3)
    sio2_slab.thick.setp(vary=True, bounds=(2, 30))
    sio2_slab.thick.name = "sio2 thickness"
    sio2_slab.rough.setp(vary=True, bounds=(0, 7))
    sio2_slab.rough.name = "sio2 roughness"
    sio2_slab.vfsolv.setp(0.1, vary=True, bounds=(0.0, 0.5))
    sio2_slab.vfsolv.name = "sio2 solvation"

    solv_roughness = Parameter(3, "bilayer/solvent roughness")
    solv_roughness.setp(vary=True, bounds=(0, 5))

    s_d2o = (
        si | sio2_slab | inner_leaflet | outer_leaflet | d2o(0, solv_roughness)
    )

    model_d2o = ReflectModel(s_d2o)

    model_d2o.scale.setp(vary=True, bounds=(0.9, 1.1))
    model_d2o.bkg.setp(vary=True, bounds=(-1e-6, 1e-6))
    objective_d2o = Objective(model_d2o, data_d2o)

    con_inner = inner_leaflet.make_constraint(objective_d2o)
    con_outer = outer_leaflet.make_constraint(objective_d2o)

    fitter = CurveFitter(objective_d2o)

    fitter.fit(
        "differential_evolution",
        constraints=(con_inner, con_outer),
        polish=False,
        popsize=10,
    )
    assert inner_leaflet.volfrac_h <= 1
    assert inner_leaflet.volfrac_t <= 1

    assert outer_leaflet.volfrac_h <= 1
    assert outer_leaflet.volfrac_t <= 1

    arr = np.array(objective_d2o.parameters)
    pc = PreparedConstraint(con_inner, arr)
    v1 = pc.violation(arr)
    apm.value = 20.0
    arr = np.array(objective_d2o.parameters)
    v2 = pc.violation(arr)
    assert (v2 > 0).all()

    assert not np.allclose(v1, v2)
