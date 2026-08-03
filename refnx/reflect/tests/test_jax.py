import os

# Set these FIRST before any other imports
os.environ["JAX_ENABLE_X64"] = "1"

import warnings
from importlib import resources
import pytest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative

import refnx
from refnx.analysis import Objective, Parameter, CurveFitter, GlobalObjective
from refnx.dataset import Data1D, ReflectDataset
from refnx.reflect import ReflectModel, SLD, LipidLeaflet, LipidLeafletGuest
from refnx.reflect.structure import overall_sld
from refnx.reflect.extra import (
    compile_objective,
    compile_model,
    make_scipy_objective,
    compile_global_objective,
    to_pymc_model,
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

    @pytest.mark.filterwarnings(
        "ignore:The _abeles_jax_ffi extension:RuntimeWarning"
    )
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

        film.real.setp(vary=True, bounds=(0.5, 2.2))
        p = Parameter(50, vary=True, bounds=(20, 100))
        t = 250 - p

        s = si | film(t, 3) | d2o(0, 3)
        model = ReflectModel(s)
        model.scale.setp(vary=True, bounds=(0.9, 1.5))

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

        check_GenerativeOp_vs_Objective(objective)

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

        # the head and tail group volume fractions
        head_vf = Parameter(0.50, "inner_head_vf", vary=True, bounds=(0.01, 1))
        tail_vf = Parameter(0.5, "tail_thickness", vary=True, bounds=(0.01, 1))

        head_thickness = v_heads / apm / head_vf
        tail_thickness = v_tails / apm / tail_vf

        # finally construct a `LipidLeaflet` object for the inner and outer leaflets.
        # Note that here the inner and outer leaflets use the same area per molecule,
        # same tail thickness, etc, but this is not necessary if the inner and outer
        # leaflets are different.
        inner_leaflet = LipidLeaflet(
            apm,
            b_heads,
            v_heads,
            head_thickness,
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
            head_thickness,
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
        model_d2o.bkg.setp(vary=True, bounds=(1e-8, 2e-6))
        objective_d2o = Objective(
            model_d2o, data_d2o, auxiliary_params=(head_vf, tail_vf)
        )

        con_inner = inner_leaflet.make_constraint(objective_d2o)
        con_outer = outer_leaflet.make_constraint(objective_d2o)

        fitter = CurveFitter(objective_d2o)

        fitter.fit(
            "differential_evolution",
            constraints=(con_inner, con_outer),
            polish=False,
            popsize=10,
        )
        assert np.all(s_d2o.slabs()[:, -1] >= 0)
        assert np.all(s_d2o.slabs()[:, -1] <= 1)

        # check that the generative op and objective.generative compare well
        # also check that changing solvent SLD changes the slab representation
        # correctly.
        obj = compile_objective(objective_d2o)
        vp = np.array(objective_d2o.varying_parameters())
        logl, _ = obj.value_and_grad(vp)
        assert_allclose(logl, objective_d2o.logl())

        # change sld of d2o
        d2o.real.setp(value=6.1234)
        vp = np.array(objective_d2o.varying_parameters())
        logl, _ = obj.value_and_grad(vp)

        assert_allclose(obj.params_to_slabs(vp), s_d2o.slabs()[:, :-1])
        assert_allclose(logl, objective_d2o.logl())

        check_GenerativeOp_vs_Objective(objective_d2o)

        # explicitly set a solvent for the structure and check that changing the
        # solvent SLD changes the slab representation correctly.
        s_d2o.solvent = d2o
        sio2_slab.vfsolv.value = 0.5
        obj = compile_objective(objective_d2o)
        vp = np.array(objective_d2o.varying_parameters())
        logl, _ = obj.value_and_grad(vp)
        assert_allclose(logl, objective_d2o.logl())

        d2o.real.setp(value=8.0)
        vp = np.array(objective_d2o.varying_parameters())
        logl, _ = obj.value_and_grad(vp)
        assert_allclose(s_d2o.slabs()[-1, 1], 8.0)
        assert_allclose(obj.params_to_slabs(vp), s_d2o.slabs()[:, :-1])

        assert_allclose(logl, objective_d2o.logl())
        check_GenerativeOp_vs_Objective(objective_d2o)

        # now assign explicit solvents for the head/tail regions and
        # check that they propagate through the params to slabs
        tail_solv = SLD(50.0)
        head_solv = SLD(30.0)
        inner_leaflet.head_solvent = head_solv
        inner_leaflet.tail_solvent = tail_solv
        tail_solv.real.setp(vary=True, bounds=(0, 100))

        obj = compile_objective(objective_d2o)

        vp = np.array(objective_d2o.varying_parameters())
        logl, _ = obj.value_and_grad(vp)
        assert_allclose(logl, objective_d2o.logl())
        assert_allclose(obj.params_to_slabs(vp), s_d2o.slabs()[:, :-1])

        tail_solv.real.value = 20.0
        vp = np.array(objective_d2o.varying_parameters())
        logl, _ = obj.value_and_grad(vp)
        assert_allclose(obj.params_to_slabs(vp), s_d2o.slabs()[:, :-1])
        assert_allclose(logl, objective_d2o.logl())
        check_GenerativeOp_vs_Objective(objective_d2o)

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
        assert_allclose(_slabs, s.slabs()[:, :-1])

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

    def test_global_objective(self):
        # checks that the global objective logl and grads are correct
        data361 = Data1D(
            Path(refnx.__file__).parent / "analysis" / "tests" / "e361r.txt"
        )
        data361.x_err = 0.05 * data361.x
        data365 = Data1D(
            Path(refnx.__file__).parent / "analysis" / "tests" / "e365r.txt"
        )
        data365.x_err = 0.05 * data365.x

        si = SLD(2.07)
        film = SLD(1.0)
        d2o = SLD(6.36)
        hdmix = SLD(3.47)
        sio2 = SLD(3.47)

        sio2_l = sio2(15, 3)
        film_l = film(200, 3)
        film_l.vfsolv.value = 0.2

        sio2_l.thick.setp(vary=True, bounds=(0, 300))
        film_l.thick.setp(vary=True, bounds=(0, 300))
        film.real.setp(vary=True, bounds=(0, 3))
        hdmix.real.setp(vary=True, bounds=(0, 5))
        d2o.real.setp(vary=True, bounds=(6.1, 6.36))

        back_rough = Parameter(3)

        s361 = si | sio2_l | film_l | d2o(0, back_rough)
        s365 = si | sio2_l | film_l | hdmix(0, back_rough)

        model361 = ReflectModel(s361)
        model365 = ReflectModel(s365)
        model361.scale.setp(vary=True, bounds=(0, 5))

        objective361 = Objective(model361, data361)
        objective365 = Objective(model365, data365)
        global_objective = GlobalObjective([objective361, objective365])

        gco = compile_global_objective(global_objective)

        d2o.real.value = 6.123
        x0 = np.array(global_objective.varying_parameters())

        v, g = gco.value_and_grad(x0)
        assert_allclose(v, global_objective.logl())

        grad = approx_derivative(global_objective.logl, x0, method="3-point")
        assert_allclose(g, grad)

        check_GenerativeOp_vs_Objective(global_objective)


def check_GenerativeOp_vs_Objective(objective, params_to_vary=None):
    # test the pymc/pytensor op. Is the generative op developed from the
    # JAX IR the same as Objective.generative
    rng = np.random.default_rng(42)
    if isinstance(objective, GlobalObjective):
        co = compile_global_objective(objective)
    else:
        co = compile_objective(objective)

    _model = to_pymc_model(objective)

    # the deterministic
    det = _model.named_vars["R_model"]

    for _ in range(20):
        if params_to_vary is None:
            idx = rng.integers(
                0, len(objective.varying_parameters()), size=None
            )
        else:
            idx = rng.choice(params_to_vary, size=None)

        p = objective.varying_parameters()[idx]
        rval = p.bounds.rvs(size=1)[0]
        p.value = rval

        init_vals = np.array(objective.varying_parameters())
        init_dct = {f"p{i}": v for i, v in enumerate(init_vals)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                assert_allclose(
                    det.eval(init_dct), objective.generative(init_vals)
                )
            except AssertionError:
                if isinstance(objective, GlobalObjective):
                    pass
                else:
                    print(p)
                    s = objective.model.structure
                    print(
                        "Are the slabs all close: ",
                        np.allclose(
                            co.params_to_slabs(init_vals), s.slabs()[:, :-1]
                        ),
                    )
                    print(co.params_to_slabs(init_vals))
                    print(s.slabs()[:, :-1])
                    print(
                        "Is the jax generative all close: ",
                        np.allclose(
                            co.generative(init_vals), objective.generative()
                        ),
                    )
                    init_vals = np.array(objective.varying_parameters())
                    init_dct = {f"p{i}": v for i, v in enumerate(init_vals)}
                    print(
                        np.allclose(
                            det.eval(init_dct), objective.generative(init_vals)
                        )
                    )
                raise
