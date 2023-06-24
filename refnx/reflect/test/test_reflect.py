import sys
import os.path
import os
import pickle
import time
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
import scipy.stats as stats
import pytest

# Before removing what appear to be unused imports think twice.
# Some of the tests use eval, which requires the imports.
import refnx.reflect._reflect as _reflect
from refnx.analysis import (
    Transform,
    Objective,
    CurveFitter,
    Parameter,
    Interval,
    Parameters,
)
from refnx.reflect import (
    SLD,
    ReflectModel,
    MixedReflectModel,
    reflectivity,
    Structure,
    Slab,
    FresnelTransform,
    choose_dq_type,
    use_reflect_backend,
)
import refnx.reflect.reflect_model as reflect_model
from refnx.dataset import ReflectDataset
from refnx._lib import MapWrapper

BACKENDS = list(reflect_model.available_backends())
# TODO re-enable pyopencl tests at some point
# pyopencl making issues on GHActions on macOS as of 11 May 2021
if sys.platform == "darwin":
    try:
        BACKENDS.remove("pyopencl")
    except ValueError:
        pass


class TestReflect:
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

        e361 = ReflectDataset(os.path.join(self.pth, "e361r.txt"))
        self.qvals361, self.rvals361, self.evals361 = (
            e361.x,
            e361.y,
            e361.y_err,
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_kernel(self, backend):
        slabs = self.structure.slabs()[..., :4]

        # test reflectivity calculation with values generated from Motofit
        with use_reflect_backend(backend) as kernel:
            calc = kernel(self.qvals, slabs)
        assert_almost_equal(calc, self.rvals)

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_noncontig_kernel(self, backend):
        # test for non-contiguous Q values
        tempq = self.qvals[0::5]
        slabs = self.structure.slabs()[..., :4]

        assert tempq.flags["C_CONTIGUOUS"] is False

        with use_reflect_backend(backend) as kernel:
            calc = kernel(tempq, slabs)
            assert_almost_equal(calc, self.rvals[0::5])

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_kernel_multithreaded(self, backend):
        slabs = self.structure.slabs()[..., :4]

        # test reflectivity calculation with values generated from Motofit
        with use_reflect_backend(backend) as kernel:
            calc = kernel(self.qvals, slabs, threads=4)
        assert_almost_equal(calc, self.rvals)

    def test_available_backends(self):
        assert "python" in BACKENDS
        assert "c" in BACKENDS
        assert "py_parratt" in BACKENDS
        assert "c_parratt" in BACKENDS
        import refnx.reflect._creflect as _creflect
        import refnx.reflect._reflect as _reflect

        assert _reflect.__file__ != _creflect.__file__

        if "cython" in BACKENDS:
            import refnx.reflect._cyreflect as _cyreflect

            assert "cython_parratt" in BACKENDS
            assert _creflect.__file__ != _cyreflect.__file__

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_first_principles(self, backend):
        # Test a first principles reflectivity calculation, rather than
        # relying on a previous calculation from Motofit code.
        # Here we only examine Fresnel reflectivity from an infinitely
        # sharp interface, we do not examine a rough surface. This is
        # tested by profile slicing in test_structure.
        def kn(q, sld_layer, sld_fronting):
            # wave vector in a given layer
            kvec = np.zeros_like(q, np.complex128)
            sld = complex(sld_layer - sld_fronting) * 1.0e-6
            kvec[:] = np.sqrt(q[:] ** 2.0 / 4.0 - 4.0 * np.pi * sld)
            return kvec

        q = np.linspace(0.001, 1.0, 1001)

        # Is the fresnel reflectivity correct?
        sld1 = 2.07
        sld2 = 6.36

        # first principles calcn
        kf = kn(q, sld1, sld1)
        kb = kn(q, sld2, sld1)
        reflectance = (kf - kb) / (kf + kb)
        reflectivity = reflectance * np.conj(reflectance)

        # now from refnx code
        struct = SLD(sld1)(0, 0) | SLD(sld2)(0, 0)
        slabs = struct.slabs()[..., :4]
        with use_reflect_backend(backend) as kernel:
            assert_allclose(kernel(q, slabs), reflectivity, rtol=1e-14)

        # reverse the direction
        kf = kn(q, sld2, sld2)
        kb = kn(q, sld1, sld2)
        reflectance = (kf - kb) / (kf + kb)
        reflectivity = reflectance * np.conj(reflectance)

        # now from refnx code
        struct = SLD(sld2)(0, 0) | SLD(sld1)(0, 0)
        slabs = struct.slabs()[..., :4]
        with use_reflect_backend(backend) as kernel:
            assert_allclose(kernel(q, slabs), reflectivity, rtol=1e-14)

    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_scale_bkg_kernel(self):
        s = self.structure.slabs()[..., :4]

        calcs = []
        for backend in BACKENDS:
            with use_reflect_backend(backend) as kernel:
                calc = kernel(self.qvals, s, scale=2.0)
                calcs.append(calc)
        for calc in calcs[1:]:
            assert_allclose(calc, calcs[0])

        calcs = []
        for backend in BACKENDS:
            with use_reflect_backend(backend) as kernel:
                calc = kernel(self.qvals, s, scale=0.5, bkg=0.1)
                calcs.append(calc)
        for calc in calcs[1:]:
            assert_allclose(calc, calcs[0])

        calcs = []
        for backend in BACKENDS:
            with use_reflect_backend(backend) as kernel:
                calc = kernel(self.qvals, s, scale=0.5, bkg=0.1, threads=2)
                calcs.append(calc)
        for calc in calcs[1:]:
            assert_allclose(calc, calcs[0])

    """
    @np.testing.decorators.knownfailure
    def test_cabeles_parallelised(self):
        # I suppose this could fail if someone doesn't have a multicore
        # computer
        if not TEST_C_REFLECT:
            return

        coefs = np.array([[0, 0, 0, 0],
                          [300, 3, 1e-3, 3],
                          [10, 3.47, 1e-3, 3],
                          [0, 6.36, 0, 3]])

        x = np.linspace(0.01, 0.2, 1000000)
        pstart = time.time()
        _creflect.abeles(x, coefs, threads=0)
        pfinish = time.time()

        sstart = time.time()
        _creflect.abeles(x, coefs, threads=1)
        sfinish = time.time()
        print(sfinish - sstart, pfinish - pstart)
        assert_(0.7 * (sfinish - sstart) > (pfinish - pstart))
    """

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_compare_kernel0(self, backend):
        # test one layer system against the python implementation
        layer0 = np.array([[0, 2.07, 0.01, 3], [0, 6.36, 0.1, 3]])
        with use_reflect_backend("python") as kernel:
            calc1 = kernel(self.qvals, layer0, scale=0.99, bkg=1e-8)

        with use_reflect_backend(backend) as kernel:
            calc2 = kernel(self.qvals, layer0, scale=0.99, bkg=1e-8)
        assert_almost_equal(calc1, calc2)

        # test a negative background
        with use_reflect_backend("python") as kernel:
            calc1 = kernel(self.qvals, layer0, scale=0.99, bkg=-5e-7)

        with use_reflect_backend(backend) as kernel:
            calc2 = kernel(self.qvals, layer0, scale=0.99, bkg=-5e-7)
        assert_almost_equal(calc1, calc2)

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_compare_kernel2(self, backend):
        # test two layer system against the python implementation
        layer2 = np.array(
            [
                [0, 2.07, 0.01, 3],
                [10, 3.47, 0.01, 3],
                [100, 1.0, 0.01, 4],
                [0, 6.36, 0.1, 3],
            ]
        )
        with use_reflect_backend("python") as kernel:
            calc1 = kernel(self.qvals, layer2, scale=0.99, bkg=1e-8)

        with use_reflect_backend(backend) as kernel:
            calc2 = kernel(self.qvals, layer2, scale=0.99, bkg=1e-8)
        assert_almost_equal(calc1, calc2)

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_kernel_absorption(self, backend):
        # https://github.com/andyfaff/refl1d_analysis/tree/master/notebooks
        q = np.linspace(0.008, 0.05, 500)
        depth = [0, 850, 0]
        rho = [2.067, 4.3, 6.0]
        irho_zero = [0.0, 0.1, 0.0]
        refnx_sigma = [np.nan, 35, 5.0]

        w_zero = np.c_[depth, rho, irho_zero, refnx_sigma]

        with use_reflect_backend("python") as kernel:
            calc1 = kernel(q, w_zero)

        with use_reflect_backend(backend) as kernel:
            calc2 = kernel(q, w_zero)
        assert_almost_equal(calc1, calc2)

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_kernel_absorption2(self, backend):
        # https://github.com/andyfaff/refl1d_analysis/tree/master/notebooks
        # this has an appreciable notch just below the critical edge
        refl1d = np.load(os.path.join(self.pth, "absorption.npy"))

        q = np.geomspace(0.005, 0.3, 201)
        depth = [0, 1200, 0]
        rho = [2.07, 4.66, 6.36]
        irho = [0, 0.016, 0]
        refnx_sigma = [np.nan, 10, 3]

        slabs = np.c_[depth, rho, irho, refnx_sigma]

        with use_reflect_backend(backend) as kernel:
            calc = kernel(q, slabs)
        assert_almost_equal(calc, refl1d[1])

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_compare_refl1d(self, backend):
        # refl1d calculated with:
        # from refl1d import abeles
        # x = np.linspace(0.005, 0.5, 1001)
        # z = abeles.refl(x / 2,
        #                 [0, 100, 200, 0],
        #                 [2.07, 3.45, 5., 6.],
        #                 irho=[0.0, 0.1, 0.01, 0],
        #                 sigma=[3, 1, 5, 0])
        # a = z.real ** 2 + z.imag ** 2

        layers = np.array(
            [
                [0, 2.07, 0, 0],
                [100, 3.45, 0.1, 3],
                [200, 5.0, 0.01, 1],
                [0, 6.0, 0, 5],
            ]
        )
        x = np.linspace(0.005, 0.5, 1001)
        refl1d = np.load(os.path.join(self.pth, "refl1d.npy"))

        with use_reflect_backend(backend) as kernel:
            calc = kernel(x, layers)
        assert_almost_equal(calc, refl1d)

    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_multilayer(self):
        x = np.geomspace(0.005, 0.5, 101)
        air = np.array([0, 0, 0, 0])
        unit_cell = np.array([[30, -2.0, 0, 3], [70, 8.0, 0, 3]])
        backing = np.array([0, 2.07, 0.0001, 3])

        def get_w(repeats=1):
            if repeats:
                filling = np.vstack([unit_cell] * repeats)
                return np.vstack([air, filling, backing])
            else:
                return np.vstack([air, backing])

        backends = list(BACKENDS)
        backends.remove("python")
        f_python = reflect_model.get_reflect_backend("python")

        for i in range(40):
            w = get_w(i)
            canonical_r = f_python(x, w)

            for backend in backends:
                with use_reflect_backend(backend) as kernel:
                    calc = kernel(x, w)
                try:
                    assert_allclose(
                        calc, canonical_r, atol=5.0e-15, rtol=8.0e-13
                    )
                except AssertionError as e:
                    print(backend, i)
                    raise e

    def test_use_reflectivity_backend(self):
        import refnx.reflect._creflect as _creflect
        import refnx.reflect._reflect as _reflect

        reflect_model.kernel = _reflect.abeles

        with use_reflect_backend("c") as f:
            assert f == _creflect.abeles
            assert reflect_model.kernel == _creflect.abeles
        assert reflect_model.kernel == _reflect.abeles

        # this shouldn't error if pyopencl is not installed
        # it should just fall back to 'c'
        reflect_model.use_reflect_backend("pyopencl")

    def test_reverse(self):
        # check that the structure reversal works.
        sio2 = SLD(3.47, name="SiO2")
        air = SLD(0, name="air")
        si = SLD(2.07, name="Si")
        structure = si | sio2(100, 3) | air(0, 2)
        structure.reverse_structure = True

        assert_equal(structure.slabs(), self.structure.slabs())

        calc = structure.reflectivity(self.qvals)
        assert_almost_equal(calc, self.rvals)

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_kernel_reshape(self, backend):
        # reflectivity should be able to deal with multidimensional input
        s = self.structure.slabs()[..., :4]
        reshaped_q = np.reshape(self.qvals, (2, 250))
        reshaped_r = self.rvals.reshape(2, 250)

        with use_reflect_backend(backend) as kernel:
            calc = kernel(reshaped_q, s)
        assert_equal(calc.shape, reshaped_r.shape)
        assert_almost_equal(calc, reshaped_r, 15)

    def test_reflectivity_model(self):
        # test reflectivity calculation with values generated from Motofit
        rff = ReflectModel(self.structure, dq=0)

        # the default for number of threads should be -1
        assert rff.threads == -1

        model = rff.model(self.qvals)
        assert_almost_equal(model, self.rvals)

    def test_mixed_reflectivity_model(self):
        # test that mixed area model works ok.

        # should be same as data generated from Motofit
        sio2 = SLD(3.47, name="SiO2")
        air = SLD(0, name="air")
        si = SLD(2.07, name="Si")

        s1 = air | sio2(100, 2) | si(0, 3)
        s2 = air | sio2(100, 2) | si(0, 3)

        mixed_model = MixedReflectModel([s1, s2], [0.4, 0.3], dq=0)
        assert_almost_equal(mixed_model(self.qvals), self.rvals * 0.7)

        # now try out the mixed model compared to sum of individual models
        # with smearing, but no background.
        s1 = air | sio2(100, 2) | si(0, 2)
        s2 = air | sio2(50, 3) | si(0, 1)

        mixed_model = MixedReflectModel([s1, s2], [0.4, 0.3], dq=5, bkg=0)
        indiv1 = ReflectModel(s1, bkg=0)
        indiv2 = ReflectModel(s2, bkg=0)

        assert_almost_equal(
            mixed_model(self.qvals),
            (0.4 * indiv1(self.qvals) + 0.3 * indiv2(self.qvals)),
        )

        # now try out the mixed model compared to sum of individual models
        # with smearing, and background.

        mixed_model.bkg.value = 1e-7
        assert_almost_equal(
            mixed_model(self.qvals),
            (0.4 * indiv1(self.qvals) + 0.3 * indiv2(self.qvals) + 1e-7),
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.filterwarnings("ignore:Using the SLOW")
    def test_parallel_calculator(self, backend):
        # test that parallel kernel work with a mapper
        q = np.linspace(0.01, 0.5, 1000).reshape(20, 50)
        p0 = np.array(
            [
                [0, 2.07, 0, 0],
                [100, 3.47, 0, 3],
                [500, -0.5, 1e-3, 3],
                [0, 6.36, 0, 3],
            ]
        )

        if backend == "pyopencl":
            # can't do pyopencl in a multiprocessing.Pool
            return

        with use_reflect_backend(backend) as kernel:
            wf = Wrapper_fn(kernel, p0)
            y = map(wf, q)
            with MapWrapper(2) as f:
                z = f(wf, q)
            assert_allclose(z, np.array(list(y)))

    def test_parallel_objective(self):
        # check that a parallel objective works without issue
        # (it could be possible that parallel evaluation fails at a higher
        #  level in e.g. emcee or in scipy.optimize.differential_evolution)
        model = self.model361
        model.threads = 2

        objective = Objective(
            model,
            (self.qvals361, self.rvals361, self.evals361),
            transform=Transform("logY"),
        )
        p0 = np.array(objective.varying_parameters())
        cov = objective.covar()
        walkers = np.random.multivariate_normal(
            np.atleast_1d(p0), np.atleast_2d(cov), size=(100)
        )
        map_logl = np.array(list(map(objective.logl, walkers)))
        map_chi2 = np.array(list(map(objective.chisqr, walkers)))

        wf = Wrapper_fn2(model.model, p0)
        map_mod = np.array(list(map(wf, walkers)))

        with MapWrapper(2) as g:
            mapw_mod = g(wf, walkers)
            mapw_logl = g(objective.logl, walkers)
            mapw_chi2 = g(objective.chisqr, walkers)
        assert_allclose(mapw_logl, map_logl)
        assert_allclose(mapw_chi2, map_chi2)
        assert_allclose(mapw_mod, map_mod)

    def test_reflectivity_fit(self):
        # a smoke test to make sure the reflectivity fit proceeds
        model = self.model361
        objective = Objective(
            model,
            (self.qvals361, self.rvals361, self.evals361),
            transform=Transform("logY"),
        )
        fitter = CurveFitter(objective)
        with np.errstate(invalid="raise"):
            fitter.fit("differential_evolution")

    def test_model_pickle(self):
        model = self.model361
        model.dq = 5.0
        model.dq_type = "constant"
        pkl = pickle.dumps(model)
        unpkl = pickle.loads(pkl)
        assert isinstance(unpkl, ReflectModel)
        for param in unpkl.parameters.flattened():
            try:
                assert isinstance(param, Parameter)
            except AssertionError:
                raise AssertionError(type(param))
        assert unpkl.dq_type == "constant"

    def test_reflectivity_emcee(self):
        model = self.model361
        model.dq = 5.0
        objective = Objective(
            model,
            (self.qvals361, self.rvals361, self.evals361),
            transform=Transform("logY"),
        )
        fitter = CurveFitter(objective, nwalkers=100)

        assert len(objective.generative().shape) == 1
        assert len(objective.residuals().shape) == 1

        res = fitter.fit("least_squares")
        res_mcmc = fitter.sample(
            steps=5, nthin=10, random_state=1, verbose=False
        )

        mcmc_val = [mcmc_result.median for mcmc_result in res_mcmc]
        assert_allclose(mcmc_val, res.x, rtol=0.05)
        # mcmc_stderr = [mcmc_result.stderr for mcmc_result in res_mcmc]
        # assert_allclose(mcmc_stderr[1:], res.stderr[1:], rtol=0.25)

    def test_smearedkernel(self):
        # test smeared reflectivity calculation with values generated from
        # Motofit (quadrature precsion order = 13)
        theoretical = np.loadtxt(
            os.path.join(self.pth, "smeared_theoretical.txt")
        )
        qvals, rvals, dqvals = np.hsplit(theoretical, 3)
        """
        the order of the quadrature precision used to create these smeared
        values in Motofit was 13.
        Do the same here
        """
        rff = ReflectModel(self.structure, quad_order=13)
        calc = rff.model(qvals.flatten(), x_err=dqvals.flatten())

        assert_almost_equal(rvals.flatten(), calc)

    def test_smearedkernel_reshape(self):
        # test smeared reflectivity calculation with values generated from
        # Motofit (quadrature precsion order = 13)
        theoretical = np.loadtxt(
            os.path.join(self.pth, "smeared_theoretical.txt")
        )
        qvals, rvals, dqvals = np.hsplit(theoretical, 3)
        """
        the order of the quadrature precision used to create these smeared
        values in Motofit was 13.
        Do the same here
        """
        reshaped_q = np.reshape(qvals, (2, 250))
        reshaped_r = np.reshape(rvals, (2, 250))
        reshaped_dq = np.reshape(dqvals, (2, 250))

        rff = ReflectModel(self.structure, quad_order=13)
        calc = rff.model(reshaped_q, x_err=reshaped_dq)

        assert_almost_equal(calc, reshaped_r)

    def test_constant_smearing(self):
        # check that constant dq/q smearing is the same as point by point
        dqvals = 0.05 * self.qvals
        rff = ReflectModel(self.structure, quad_order="ultimate")
        calc = rff.model(self.qvals, x_err=dqvals)

        rff.dq = 5.0
        calc2 = rff.model(self.qvals)

        assert_allclose(calc, calc2, rtol=0.011)

    def test_resolution_kernel(self):
        # check that resolution kernel works, use constant dq/q of 5% as
        # comparison
        slabs = self.structure361.slabs()[:, :4]
        npnts = 1000
        q = np.linspace(0.005, 0.3, npnts)

        # use constant dq/q for comparison
        const_R = reflectivity(
            q,
            slabs,
            scale=1.01,
            bkg=1e-6,
            dq=0.05 * q,
            quad_order=101,
            threads=-1,
        )

        # lets create a kernel.
        kernel = np.zeros((npnts, 2, 501), float)
        sd = 0.05 * q / (2 * np.sqrt(2 * np.log(2)))
        for i in range(npnts):
            kernel[i, 0, :] = np.linspace(
                q[i] - 3.5 * sd[i], q[i] + 3.5 * sd[i], 501
            )
            kernel[i, 1, :] = stats.norm.pdf(
                kernel[i, 0, :], loc=q[i], scale=sd[i]
            )

        kernel_R = reflectivity(q, slabs, scale=1.01, bkg=1e-6, dq=kernel)

        assert_allclose(kernel_R, const_R, rtol=0.002)

    def test_sld_profile(self):
        # test SLD profile with SLD profile from Motofit.
        np.seterr(invalid="raise")
        profile = np.loadtxt(os.path.join(self.pth, "sld_theoretical_R.txt"))
        z, rho = np.split(profile, 2)

        rff = ReflectModel(self.structure)
        z, myrho = rff.structure.sld_profile(z.flatten())
        assert_almost_equal(myrho, rho.flatten())

    def test_modelvals_degenerate_layers(self):
        # try fitting dataset with a deposited layer split into two degenerate
        # layers
        fname = os.path.join(self.pth, "c_PLP0011859_q.txt")
        dataset = ReflectDataset(fname)

        sio2 = SLD(3.47, name="SiO2")
        si = SLD(2.07, name="Si")
        d2o = SLD(6.36, name="D2O")
        polymer = SLD(2.0, name="polymer")

        sio2_l = sio2(30, 3)
        polymer_l = polymer(125, 3)

        structure = si | sio2_l | polymer_l | polymer_l | d2o(0, 3)

        polymer_l.thick.setp(value=125, vary=True, bounds=(0, 250))
        polymer_l.rough.setp(value=4, vary=True, bounds=(0, 8))
        structure[-1].rough.setp(vary=True, bounds=(0, 6))
        sio2_l.rough.setp(value=3.16, vary=True, bounds=(0, 8))

        model = ReflectModel(structure, bkg=2e-6)
        objective = Objective(
            model, dataset, use_weights=False, transform=Transform("logY")
        )

        model.scale.setp(vary=True, bounds=(0, 2))
        model.bkg.setp(vary=True, bounds=(0, 8e-6))

        slabs = structure.slabs()
        assert_equal(slabs[2, 0:2], slabs[3, 0:2])
        assert_equal(slabs[2, 3], slabs[3, 3])
        assert_equal(slabs[1, 3], sio2_l.rough.value)

        f = CurveFitter(objective)
        f.fit(method="differential_evolution", seed=1, maxiter=3)

        slabs = structure.slabs()
        assert_equal(slabs[2, 0:2], slabs[3, 0:2])
        assert_equal(slabs[2, 3], slabs[3, 3])

    def test_resolution_speed_comparator(self):
        fname = os.path.join(self.pth, "c_PLP0011859_q.txt")
        dataset = ReflectDataset(fname)

        sio2 = SLD(3.47, name="SiO2")
        si = SLD(2.07, name="Si")
        d2o = SLD(6.36, name="D2O")
        polymer = SLD(2.0, name="polymer")

        sio2_l = sio2(30, 3)
        polymer_l = polymer(125, 3)

        dx = dataset.x_err
        structure = si | sio2_l | polymer_l | polymer_l | d2o(0, 3)
        model = ReflectModel(structure, bkg=2e-6, dq_type="constant")
        objective = Objective(
            model, dataset, use_weights=False, transform=Transform("logY")
        )

        # check that choose_resolution_approach doesn't change state
        # of model
        fastest_method = choose_dq_type(objective)
        assert model.dq_type == "constant"
        assert_equal(dx, objective.data.x_err)

        # check that the comparison worked
        const_time = time.time()
        for i in range(1000):
            objective.generative()
        const_time = time.time() - const_time

        model.dq_type = "pointwise"
        point_time = time.time()
        for i in range(1000):
            objective.generative()
        point_time = time.time() - point_time

        if fastest_method == "pointwise":
            assert point_time < const_time
        elif fastest_method == "constant":
            assert const_time < point_time

        # check that we could use the function to setup a reflectmodel
        ReflectModel(structure, bkg=2e-6, dq_type=choose_dq_type(objective))

    def test_mixed_model(self):
        # test for MixedReflectModel
        air = SLD(0, name="air")
        sio2 = SLD(3.47, name="SiO2")
        si = SLD(2.07, name="Si")

        structure1 = air | sio2(100, 2) | si(0, 3)
        structure2 = air | sio2(50, 3) | si(0, 5)

        # this is out theoretical calculation
        mixed_model_y = 0.4 * structure1.reflectivity(self.qvals)
        mixed_model_y += 0.6 * structure2.reflectivity(self.qvals)

        mixed_model = MixedReflectModel(
            [structure1, structure2], [0.4, 0.6], bkg=0, dq=0
        )

        assert_equal(mixed_model.scales, np.array([0.4, 0.6]))
        assert mixed_model.dq.value == 0

        assert_allclose(mixed_model_y, mixed_model(self.qvals), atol=1e-13)

        # test repr of MixedReflectModel
        q = repr(mixed_model)
        r = eval(q)
        assert_equal(r.scales, np.array([0.4, 0.6]))
        assert r.dq.value == 0

        assert_allclose(mixed_model_y, r(self.qvals), atol=1e-13)

    def test_pnr(self):
        # test pnr calculation
        q = np.linspace(0.01, 0.3, 1001)

        # use for spin channel PNR calculation
        players = np.array(
            [[0, 0, 0, 0, 0], [100, 3, 0, 1, 0], [0, 4, 0, 0, 0]]
        )

        # use for NSF calculation with kernel
        pp_layers = np.array([[0, 0, 0, 0], [100, 4.0, 0, 0], [0, 4, 0, 0]])

        mm_layers = np.array([[0, 0, 0, 0], [100, 2, 0, 0], [0, 4, 0, 0]])

        r = _reflect.pnr(q, players)

        pp = _reflect.abeles(q, pp_layers)
        mm = _reflect.abeles(q, mm_layers)

        assert_allclose(r[0], pp)
        assert_allclose(r[1], mm)

    def test_repr_reflect_model(self):
        p = SLD(0.0)
        q = SLD(2.07)
        s = p(0, 0) | q(0, 3)
        model = ReflectModel(s, scale=0.99, bkg=1e-8, q_offset=0.002)
        r = eval(repr(model))

        x = np.linspace(0.005, 0.3, 1000)
        assert_equal(r(x), model(x))

    def test_q_offset(self):
        p = SLD(0.0)
        q = SLD(2.07)
        s = p(0, 0) | q(0, 3)
        model = ReflectModel(s, scale=0.99, bkg=1e-8, q_offset=0.002)
        model2 = ReflectModel(s, scale=0.99, bkg=1e-8, q_offset=0.0)

        x = np.linspace(0.01, 0.2, 3)
        assert_equal(model(x - 0.002), model2(x))

    def test_FresnelTransform(self):
        t = FresnelTransform(2.07, 6.36, dq=5)

        slabs = np.array([[0, 2.07, 0, 0], [0, 6.36, 0, 0]])

        q = np.linspace(0.01, 1.0, 1000)
        r = reflectivity(q, slabs, dq=5)

        rt, et = t(q, r)
        assert_almost_equal(rt, 1.0)

        # test errors are transformed
        rt, et = t(q, r, y_err=1.0)
        assert_almost_equal(et, 1.0 / r)

        # check that you can use an SLD object
        t = FresnelTransform(SLD(2.07), SLD(6.36), dq=5)
        rt, et = t(q, r)
        assert_almost_equal(rt, 1.0)

        # reconstitute a repr'd transform and check it works
        s = repr(t)
        st = eval(s)
        rt, et = st(q, r)
        assert_almost_equal(rt, 1.0)


class Wrapper_fn:
    def __init__(self, fn, w):
        self.fn = fn
        self.w = w

    def __call__(self, x):
        assert len(x.shape) == 1
        return self.fn(x, self.w, threads=1)


class Wrapper_fn2:
    def __init__(self, fn, w):
        self.fn = fn
        self.w = w

    def __call__(self, x):
        assert len(x.shape) == 1
        return self.fn(x, self.w)
