import os.path
import os
import pickle

try:
    import refnx.reflect._creflect as _creflect
    HAVE_CREFLECT = True
except ImportError:
    HAVE_CREFLECT = False

import refnx.reflect._reflect as _reflect
from refnx.analysis import (Transform, Objective,
                            CurveFitter, Parameter, Model)
from refnx.reflect import (SLD, Slab, ReflectModel, MixedReflectModel)
from refnx.dataset import ReflectDataset

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_allclose)


# if REQUIRE_C is specified then definitely test C plugins
REQUIRE_C = os.environ.get('REQUIRE_C', 0)
TEST_C_REFLECT = HAVE_CREFLECT or REQUIRE_C


class TestReflect(object):

    def setup_method(self):
        self.pth = os.path.dirname(os.path.abspath(__file__))

        sio2 = SLD(3.47, name='SiO2')
        air = SLD(0, name='air')
        si = SLD(2.07, name='Si')
        d2o = SLD(6.36, name='D2O')
        polymer = SLD(1, name='polymer')

        self.structure = air | sio2(100, 2) | si(0, 3)

        theoretical = np.loadtxt(os.path.join(self.pth, 'theoretical.txt'))
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

        e361 = ReflectDataset(os.path.join(self.pth, 'e361r.txt'))
        self.qvals361, self.rvals361, self.evals361 = (e361.x,
                                                       e361.y,
                                                       e361.y_err)

    def test_abeles(self):
        # test reflectivity calculation with values generated from Motofit
        calc = self.structure.reflectivity(self.qvals)
        assert_almost_equal(calc, self.rvals)

    def test_c_abeles(self):
        if TEST_C_REFLECT:
            # test reflectivity calculation with values generated from Motofit
            calc = _creflect.abeles(self.qvals, self.structure.slabs[..., :4])
            assert_almost_equal(calc, self.rvals)

            # test for non-contiguous Q values
            tempq = self.qvals[0::5]
            assert_(tempq.flags['C_CONTIGUOUS'] is False)
            calc = _creflect.abeles(tempq, self.structure.slabs[..., :4])
            assert_almost_equal(calc, self.rvals[0::5])

    def test_py_abeles(self):
        # test reflectivity calculation with values generated from Motofit
        calc = _reflect.abeles(self.qvals, self.structure.slabs[..., :4])
        assert_almost_equal(calc, self.rvals)

    def test_compare_c_py_abeles(self):
        # test python and c are equivalent
        # but not the same file
        s = self.structure.slabs[..., :4]

        if not TEST_C_REFLECT:
            return
        assert_(_reflect.__file__ != _creflect.__file__)

        calc1 = _reflect.abeles(self.qvals, s)
        calc2 = _creflect.abeles(self.qvals, s)
        assert_almost_equal(calc1, calc2)
        calc1 = _reflect.abeles(self.qvals, s, scale=2.)
        calc2 = _creflect.abeles(self.qvals, s, scale=2.)
        assert_almost_equal(calc1, calc2)
        calc1 = _reflect.abeles(self.qvals, s, scale=0.5,
                                bkg=0.1)
        # threads = 1 is a non-threaded implementation
        calc2 = _creflect.abeles(self.qvals, s, scale=0.5,
                                 bkg=0.1, threads=1)
        # threads = 2 forces the calculation to go through multithreaded calcn,
        # even on single core processor
        calc3 = _creflect.abeles(self.qvals, s, scale=0.5,
                                 bkg=0.1, threads=2)
        assert_almost_equal(calc1, calc2)
        assert_almost_equal(calc1, calc3)

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

    def test_compare_c_py_abeles0(self):
        # test two layer system
        if not TEST_C_REFLECT:
            return
        layer0 = np.array([[0, 2.07, 0.01, 3],
                           [0, 6.36, 0.1, 3]])
        calc1 = _reflect.abeles(self.qvals, layer0, scale=0.99, bkg=1e-8)
        calc2 = _creflect.abeles(self.qvals, layer0, scale=0.99, bkg=1e-8)
        assert_almost_equal(calc1, calc2)

    def test_compare_c_py_abeles2(self):
        # test two layer system
        if not TEST_C_REFLECT:
            return
        layer2 = np.array([[0, 2.07, 0.01, 3],
                           [10, 3.47, 0.01, 3],
                           [100, 1.0, 0.01, 4],
                           [0, 6.36, 0.1, 3]])
        calc1 = _reflect.abeles(self.qvals, layer2, scale=0.99, bkg=1e-8)
        calc2 = _creflect.abeles(self.qvals, layer2, scale=0.99, bkg=1e-8)
        assert_almost_equal(calc1, calc2)

    def test_reverse(self):
        # check that the structure reversal works.
        sio2 = SLD(3.47, name='SiO2')
        air = SLD(0, name='air')
        si = SLD(2.07, name='Si')
        structure = si | sio2(100, 3) | air(0, 2)
        structure.reverse_structure = True

        assert_equal(structure.slabs, self.structure.slabs)

        calc = structure.reflectivity(self.qvals)
        assert_almost_equal(calc, self.rvals)

    def test_c_abeles_reshape(self):
        # c reflectivity should be able to deal with multidimensional input
        if not TEST_C_REFLECT:
            return
        s = self.structure.slabs[..., :4]

        reshaped_q = np.reshape(self.qvals, (2, 250))
        reshaped_r = self.rvals.reshape(2, 250)
        calc = _creflect.abeles(reshaped_q, s)
        assert_equal(reshaped_r.shape, calc.shape)
        assert_almost_equal(reshaped_r, calc, 15)

    def test_abeles_reshape(self):
        # reflectivity should be able to deal with multidimensional input
        s = self.structure.slabs[..., :4]

        reshaped_q = np.reshape(self.qvals, (2, 250))
        reshaped_r = self.rvals.reshape(2, 250)
        calc = _reflect.abeles(reshaped_q, s)
        assert_equal(reshaped_r.shape, calc.shape)
        assert_almost_equal(reshaped_r, calc, 15)

    def test_reflectivity_model(self):
        # test reflectivity calculation with values generated from Motofit
        rff = ReflectModel(self.structure, dq=0)
        model = rff.model(self.qvals)
        assert_almost_equal(model, self.rvals)

    def test_mixed_reflectivity_model(self):
        # test that mixed area model works ok.

        # should be same as data generated from Motofit
        sio2 = SLD(3.47, name='SiO2')
        air = SLD(0, name='air')
        si = SLD(2.07, name='Si')

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

        assert_almost_equal(mixed_model(self.qvals),
                            (0.4 * indiv1(self.qvals) +
                             0.3 * indiv2(self.qvals)))

        # now try out the mixed model compared to sum of individual models
        # with smearing, and background.

        mixed_model.bkg.value = 1e-7
        assert_almost_equal(mixed_model(self.qvals),
                            (0.4 * indiv1(self.qvals) +
                             0.3 * indiv2(self.qvals) +
                             1e-7))

    def test_reflectivity_fit(self):
        # a smoke test to make sure the reflectivity fit proceeds
        model = self.model361
        objective = Objective(model,
                              (self.qvals361, self.rvals361, self.evals361),
                              transform=Transform('logY'))
        fitter = CurveFitter(objective)
        with np.errstate(invalid='raise'):
            fitter.fit('differential_evolution')

    def test_model_pickle(self):
        model = self.model361
        model.dq = 5.
        pkl = pickle.dumps(model)
        unpkl = pickle.loads(pkl)
        assert_(isinstance(unpkl, ReflectModel))
        for param in unpkl.parameters.flattened():
            try:
                assert_(isinstance(param, Parameter))
            except AssertionError:
                raise AssertionError(type(param))

    def test_reflectivity_emcee(self):
        model = self.model361
        model.dq = 5.
        objective = Objective(model,
                              (self.qvals361, self.rvals361, self.evals361),
                              transform=Transform('logY'))
        fitter = CurveFitter(objective, nwalkers=100)

        assert_(len(objective.generative().shape) == 1)
        assert_(len(objective.residuals().shape) == 1)

        res = fitter.fit('least_squares')
        res_mcmc = fitter.sample(steps=5, nthin=10, random_state=1,
                                 verbose=False)

        mcmc_val = [mcmc_result.median for mcmc_result in res_mcmc]
        assert_allclose(mcmc_val, res.x, rtol=0.05)
        # mcmc_stderr = [mcmc_result.stderr for mcmc_result in res_mcmc]
        # assert_allclose(mcmc_stderr[1:], res.stderr[1:], rtol=0.25)

    def test_smearedabeles(self):
        # test smeared reflectivity calculation with values generated from
        # Motofit (quadrature precsion order = 13)
        theoretical = np.loadtxt(os.path.join(self.pth,
                                              'smeared_theoretical.txt'))
        qvals, rvals, dqvals = np.hsplit(theoretical, 3)
        '''
        the order of the quadrature precision used to create these smeared
        values in Motofit was 13.
        Do the same here
        '''
        rff = ReflectModel(self.structure, quad_order=13)
        calc = rff.model(qvals.flatten(), x_err=dqvals.flatten())

        assert_almost_equal(rvals.flatten(), calc)

    def test_smearedabeles_reshape(self):
        # test smeared reflectivity calculation with values generated from
        # Motofit (quadrature precsion order = 13)
        theoretical = np.loadtxt(os.path.join(self.pth,
                                              'smeared_theoretical.txt'))
        qvals, rvals, dqvals = np.hsplit(theoretical, 3)
        '''
        the order of the quadrature precision used to create these smeared
        values in Motofit was 13.
        Do the same here
        '''
        reshaped_q = np.reshape(qvals, (2, 250))
        reshaped_r = np.reshape(rvals, (2, 250))
        reshaped_dq = np.reshape(dqvals, (2, 250))

        rff = ReflectModel(self.structure, quad_order=13)
        calc = rff.model(reshaped_q, x_err=reshaped_dq)

        assert_almost_equal(calc, reshaped_r)

    def test_constant_smearing(self):
        # check that constant dq/q smearing is the same as point by point
        dqvals = 0.05 * self.qvals
        rff = ReflectModel(self.structure, quad_order='ultimate')
        calc = rff.model(self.qvals, x_err=dqvals)

        rff.dq = 5.
        calc2 = rff.model(self.qvals)

        assert_allclose(calc, calc2, rtol=0.011)

    def test_sld_profile(self):
        # test SLD profile with SLD profile from Motofit.
        np.seterr(invalid='raise')
        profile = np.loadtxt(os.path.join(self.pth, 'sld_theoretical_R.txt'))
        z, rho = np.split(profile, 2)

        rff = ReflectModel(self.structure)
        z, myrho = rff.structure.sld_profile(z.flatten())
        assert_almost_equal(myrho, rho.flatten())

    def test_modelvals_degenerate_layers(self):
        # try fitting dataset with a deposited layer split into two degenerate
        # layers
        fname = os.path.join(self.pth, 'c_PLP0011859_q.txt')
        dataset = ReflectDataset(fname)

        sio2 = SLD(3.47, name='SiO2')
        si = SLD(2.07, name='Si')
        d2o = SLD(6.36, name='D2O')
        polymer = SLD(2., name='polymer')

        sio2_l = sio2(30, 3)
        polymer_l = polymer(125, 3)

        structure = (si | sio2_l | polymer_l | polymer_l | d2o(0, 3))

        polymer_l.thick.setp(value=125, vary=True, bounds=(0, 250))
        polymer_l.rough.setp(value=4, vary=True, bounds=(0, 8))
        structure[-1].rough.setp(vary=True, bounds=(0, 6))
        sio2_l.rough.setp(value=3.16, vary=True, bounds=(0, 8))

        model = ReflectModel(structure, bkg=2e-6)
        objective = Objective(model,
                              dataset,
                              use_weights=False,
                              transform=Transform('logY'))

        model.scale.setp(vary=True, bounds=(0, 2))
        model.bkg.setp(vary=True, bounds=(0, 8e-6))

        slabs = structure.slabs
        assert_equal(slabs[2, 0:2], slabs[3, 0:2])
        assert_equal(slabs[2, 3], slabs[3, 3])
        assert_equal(slabs[1, 3], sio2_l.rough.value)

        f = CurveFitter(objective)
        f.fit(method='differential_evolution', seed=1)

        slabs = structure.slabs
        assert_equal(slabs[2, 0:2], slabs[3, 0:2])
        assert_equal(slabs[2, 3], slabs[3, 3])

    def test_mixed_model(self):
        # test for MixedReflectModel
        air = SLD(0, name='air')
        sio2 = SLD(3.47, name='SiO2')
        si = SLD(2.07, name='Si')

        structure1 = air | sio2(100, 2) | si(0, 3)
        structure2 = air | sio2(50, 3) | si(0, 5)

        # this is out theoretical calculation
        mixed_model_y = 0.4 * structure1.reflectivity(self.qvals)
        mixed_model_y += 0.6 * structure2.reflectivity(self.qvals)

        mixed_model = MixedReflectModel([structure1, structure2], [0.4, 0.6],
                                        bkg=0, dq=0)

        assert_equal(mixed_model.scales, np.array([0.4, 0.6]))
        assert_(mixed_model.dq.value == 0)

        assert_equal(mixed_model_y, mixed_model(self.qvals))
