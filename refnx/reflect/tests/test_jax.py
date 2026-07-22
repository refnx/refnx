from importlib import resources
import pytest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose

import refnx
from refnx.analysis import Objective, Parameter, CurveFitter
from refnx.dataset import Data1D, ReflectDataset
from refnx.reflect import ReflectModel, SLD, LipidLeaflet, LipidLeafletGuest
from refnx.reflect.structure import overall_sld
from refnx.reflect.extra import (
    compile_objective,
    compile_model,
    make_scipy_objective,
)

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
        data.x_err *= 2.3548

        self.objective = Objective(model, data)

    def test_compile_objective(self):
        # Obtain the negative log-likelihood (nll) from the compiled objective
        # By looking at the nll we're implicitly checking resolution smearing,
        # nll calculation, etc
        obj = compile_objective(self.objective)
        vg = obj.value_and_grad
        logl, grad = vg(np.array(self.objective.varying_parameters()))
        assert_allclose(-logl, self.objective.nll())

    def test_solvation_reverse(self):
        # experiment with solvation and reversing structure and check that
        # solvation is occurring properly.
        s = self.structure
        s[1].vfsolv.value = 0.4
        s[1].rough.value = 1
        s[2].rough.value = 2
        s[3].rough.value = 3

        co = compile_objective(self.objective)
        pars = np.array(self.objective.varying_parameters())
        _slabs = co.params_to_slabs(pars)
        assert_allclose(_slabs, s.slabs()[:, :-1])

        # reverse model and check
        s.reverse_structure = True
        co = compile_objective(self.objective)
        pars = np.array(self.objective.varying_parameters())
        _slabs = co.params_to_slabs(pars)
        assert_allclose(_slabs, s.slabs()[:, :-1])

        # now set specific solvent
        new_solv = SLD(1.2345 + 5.122j)
        s.solvent = new_solv
        s.reverse_structure = False

        co = compile_objective(self.objective)
        pars = np.array(self.objective.varying_parameters())
        _slabs = co.params_to_slabs(pars)
        assert_allclose(_slabs, s.slabs()[:, :-1])

    def test_auxiliary_parameters(self):
        data = Data1D(
            Path(refnx.__file__).parent / "analysis" / "tests" / "e361r.txt"
        )
        data.x_err = 0.05 * data.x

        si = SLD(2.07)
        film = SLD(1.0)
        d2o = SLD(6.36)

        film.real.setp(vary=True)
        p = Parameter(50, vary=True)
        t = 250 - p

        s = si | film(t, 3) | d2o(0, 3)
        model = ReflectModel(s)
        model.scale.setp(vary=True)

        objective = Objective(model, data, auxiliary_params=(p,))
        pars = np.array(objective.varying_parameters())

        p.value = 49
        nll49 = objective.nll()
        p.value = 50
        nll50 = objective.nll()
        p.value = 48
        nll48 = objective.nll()

        c = compile_objective(objective)

        nll_fn, grad_fn = make_scipy_objective(c)
        assert_allclose(nll_fn(pars), nll50)

        pars[1] = 49.0
        assert_allclose(nll_fn(pars), nll49)

        pars[1] = 48.0
        assert_allclose(nll_fn(pars), nll48)

    def test_lipid(self):
        pth = resources.files(refnx.analysis) / "tests"

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
            si
            | sio2_slab
            | inner_leaflet
            | outer_leaflet
            | d2o(0, solv_roughness)
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
        obj = compile_objective(objective_d2o)
        logl, _ = obj.value_and_grad(
            np.array(objective_d2o.varying_parameters())
        )
        assert_allclose(logl, objective_d2o.logl())

    def test_lipidleaflet_guest(self):
        b_h = 6.01e-4
        V_h = 319.0
        b_t = -2.92e-4
        V_t = 782.0
        APM = 60.0
        thick_h = 9.0
        thick_t = 14.0

        phi_guest_t = Parameter(0.1)
        sld_guest = SLD(7.6)
        with pytest.warns(RuntimeWarning):
            leaflet = LipidLeafletGuest(
                APM,
                b_h,
                V_h,
                thick_h,
                b_t,
                V_t,
                thick_t,
                2,
                3,
                0,
                phi_guest_t,
                sld_guest,
            )

        # check slab representation
        si = SLD(2.07)
        d2o = SLD(6.36)
        s = si | leaflet | d2o(0, 3)

        model = ReflectModel(s)
        model.scale.setp(vary=True, bounds=(0, 5))

        cm = compile_model(model)
        _slabs = cm.params_to_slabs(np.array([1.0]))
        assert_allclose(_slabs[:, 1], s.slabs()[:, 1], rtol=1e-10)

    def test_lipidleafletguest_solvent_specified(self):
        phi_guest_t = Parameter(0.1)
        sld_guest = SLD(7.6)
        sld_solvent = SLD(5.55)
        b_h = 6.01e-4
        V_h = 319.0
        b_t = -2.92e-4
        V_t = 782.0
        APM = 60.0
        thick_h = 9.0
        thick_t = 14.0

        with pytest.warns(RuntimeWarning):
            leaflet = LipidLeafletGuest(
                APM,
                b_h,
                V_h,
                thick_h,
                b_t,
                V_t,
                thick_t,
                2,
                3,
                0,
                phi_guest_t,
                sld_guest,
                head_solvent=sld_solvent,
                tail_solvent=sld_solvent,
            )
        # check slab representation
        si = SLD(2.07)
        d2o = SLD(6.36)
        s = si | leaflet | d2o(0, 3)

        model = ReflectModel(s)
        model.scale.setp(vary=True, bounds=(0, 5))

        cm = compile_model(model)
        _slabs = cm.params_to_slabs(np.array([1.0]))
        assert_allclose(_slabs[:, 1], s.slabs()[:, 1], rtol=1e-10)
