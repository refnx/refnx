import pickle

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_,
    assert_allclose,
)
from scipy.stats import cauchy
from refnx._lib import flatten
from refnx.reflect import (
    SLD,
    Structure,
    Spline,
    Slab,
    Stack,
    Erf,
    Linear,
    Exponential,
    Interface,
    MaterialSLD,
    MixedSlab,
)
from refnx.reflect.structure import _profile_slicer
from refnx.analysis import Parameter, Interval, Parameters
from refnx.analysis.parameter import _BinaryOp


class TestStructure:
    def setup_method(self):
        self.air = SLD(0, name="air")
        self.sio2 = SLD(3.47, name="sio2")
        self.d2o = SLD(6.36, name="d2o")
        self.h2o = SLD(-0.56, name="h2o")
        self.s = self.air | self.sio2(100, 5) | self.d2o(0, 4)

    def test_structure_construction(self):
        # structures are constructed by or-ing slabs
        # test that the slab representation is correct
        assert_equal(
            self.s.slabs(),
            np.array(
                [[0, 0, 0, 0, 0], [100, 3.47, 0, 5, 0], [0, 6.36, 0, 4, 0]]
            ),
        )

        self.s[1] = SLD(3.47 + 1j, name="sio2")(100, 5)
        self.s[1].vfsolv.value = 0.9

        oldpars = len(list(flatten(self.s.parameters)))

        # slabs have solvent penetration
        self.s.solvent = SLD(5 + 1.2j)
        sld = 5 * 0.9 + 0.1 * 3.47
        sldi = 1 * 0.1 + 0.9 * 1.2
        assert_almost_equal(
            self.s.slabs(),
            np.array(
                [[0, 0, 0, 0, 0], [100, sld, sldi, 5, 0.9], [0, 6.36, 0, 4, 0]]
            ),
        )

        # when the structure._solvent is not None, but an SLD object, then
        # it's number of parameters should increase by 2.
        newpars = len(list(flatten(self.s.parameters)))
        assert_equal(newpars, oldpars + 2)

        # by default solvation is done by backing medium
        self.s.solvent = None
        sld = 6.36 * 0.9 + 0.1 * 3.47
        sldi = 1 * 0.1
        assert_almost_equal(
            self.s.slabs(),
            np.array(
                [[0, 0, 0, 0, 0], [100, sld, sldi, 5, 0.9], [0, 6.36, 0, 4, 0]]
            ),
        )

        # by default solvation is done by backing medium, except when structure
        # is reversed
        self.s.reverse_structure = True
        sld = 0 * 0.9 + 0.1 * 3.47
        sldi = 0 * 0.9 + 1 * 0.1
        assert_almost_equal(
            self.s.slabs(),
            np.array(
                [[0, 6.36, 0, 0, 0], [100, sld, sldi, 4, 0.9], [0, 0, 0, 5, 0]]
            ),
        )

    def test_interface(self):
        # can we set the interface property correctly
        c = self.sio2(10, 3)
        assert c.interfaces is None

        c.interfaces = Erf()
        assert isinstance(c.interfaces, Erf)

        c.interfaces = [Erf()]
        assert isinstance(c.interfaces, Erf)

        c.interfaces = None
        assert c.interfaces is None

        import pytest

        with pytest.raises(ValueError):
            c.interfaces = [1]

        # because len(c.slabs()) = 1
        with pytest.raises(ValueError):
            c.interfaces = [Erf(), Erf()]

    def test_mixed_slab(self):
        m = MixedSlab(
            10.0,
            [1, 2 + 0.1j, 3.0 + 1j],
            [0.1, 0.2, 0.3],
            10.0,
            vfsolv=0.1,
            interface=Linear(),
            name="pop",
        )
        slabs = m.slabs()
        assert_allclose(slabs[0, 0], 10.0)

        assert_allclose(slabs[0, 1], 2.3333333333333333)
        assert_allclose(slabs[0, 2], 0.5333333333333333)
        assert_allclose(slabs[0, 3], 10.0)
        assert_allclose(slabs[0, 4], 0.1)
        assert_equal(float(m.vfsolv), 0.1)
        assert m.name == "pop"

        # test the repr
        q = eval(repr(m))
        slabs = q.slabs()
        assert_allclose(slabs[0, 0], 10.0)

        assert_allclose(slabs[0, 1], 2.3333333333333333)
        assert_allclose(slabs[0, 2], 0.5333333333333333)
        assert_allclose(slabs[0, 3], 10.0)
        assert_allclose(slabs[0, 4], 0.1)

        assert_equal(float(q.vfsolv), 0.1)

    def test_micro_slab(self):
        # test micro-slab representation by calculating reflectivity from a
        # structure with default interfacial profiles for all the components.
        # Then specify an Erf interface for the slab and check that the
        # reflectivity signal is the same.
        sio2 = self.sio2(100, 5)
        d2o = self.d2o(0, 4)

        s = self.air | sio2 | d2o
        s.contract = -1
        q = np.linspace(0.01, 0.5, 101)
        reflectivity = s.reflectivity(q)

        sio2.interfaces = Erf()
        d2o.interfaces = Erf()

        micro_slab_reflectivity = s.reflectivity(q)

        # Should be within 1%
        # How close the micro-slicing is to the Nevot-Croce is going to
        # depend on the exact system you look at, and what slice thickness
        # is used.
        assert_allclose(micro_slab_reflectivity, reflectivity, rtol=0.01)

        # test out user defined roughness type
        class Cauchy(Interface):
            def __call__(self, x, loc=0, scale=1):
                return cauchy.cdf(x, loc=loc, scale=scale)

        c = Cauchy()
        sio2.interfaces = c
        s.reflectivity(q)

        # imaginary part of micro slab should be calculated in same way as
        # real part
        fronting = SLD(1 + 1j)
        layer = SLD(4 + 4j)
        backing = SLD(6 + 6j)
        s = fronting | layer(100, 4) | backing(0, 4)
        s[1].interfaces = Erf()
        s[-1].interfaces = Erf()
        slabs = s.slabs()
        assert_almost_equal(slabs[:, 1], slabs[:, 2])

    def test_pickle(self):
        # need to be able to pickle and unpickle structure
        pkl = pickle.dumps(self.s)
        unpkl = pickle.loads(pkl)
        assert_(isinstance(unpkl, Structure))
        for param in unpkl.parameters.flattened():
            assert_(isinstance(param, Parameter))
        assert hasattr(unpkl, "_solvent")

    def test_sld_profile(self):
        # check that it runs
        z, sld_profile = self.s.sld_profile()
        assert_equal(np.size(z), 500)

        z, sld_profile = self.s.sld_profile(max_delta_z=0.251)
        delta_z = np.ediff1d(z)
        assert delta_z[0] <= 0.251

        z, sld_profile = self.s.sld_profile(np.linspace(-100, 100, 100))
        assert_equal(min(z), -100)
        assert_equal(max(z), 100)

    def test_reflectivity(self):
        q = np.linspace(0.005, 0.3, 200)
        self.s.reflectivity(q)

    def test_repr_sld(self):
        p = SLD(5 + 1j, name="pop")
        assert_equal(float(p.real), 5)
        assert_equal(float(p.imag), 1)
        print(repr(p))
        q = eval(repr(p))
        assert_equal(float(q.real), 5)
        assert_equal(float(q.imag), 1)

    def test_repr_materialsld(self):
        p = MaterialSLD("SiO2", density=2.2, name="silica")
        sldc = complex(p)
        assert_allclose(sldc.real, 3.4752690258246504)
        assert_allclose(sldc.imag, 1.0508799522721932e-05)
        print(repr(p))
        q = eval(repr(p))
        sldc = complex(q)
        assert_allclose(sldc.real, 3.4752690258246504)
        assert_allclose(sldc.imag, 1.0508799522721932e-05)

    def test_materialsld(self):
        p = MaterialSLD("SiO2", density=2.2, name="silica")
        sldc = complex(p)
        assert_allclose(sldc.real, 3.4752690258246504)
        assert_allclose(sldc.imag, 1.0508799522721932e-05)
        assert p.probe == "neutron"

        # is X-ray SLD correct?
        p.wavelength = 1.54
        p.probe = "x-ray"
        sldc = complex(p)
        assert_allclose(sldc.real, 18.864796064009866)
        assert_allclose(sldc.imag, 0.2436013463223236)

        assert len(p.parameters) == 1
        assert p.formula == "SiO2"

        # the density value should change the SLD
        p.probe = "neutron"
        p.density.value = 4.4
        sldc = complex(p)
        assert_allclose(sldc.real, 3.4752690258246504 * 2)
        assert_allclose(sldc.imag, 1.0508799522721932e-05 * 2)

        # should be able to make a Slab from MaterialSLD
        slab = p(10, 3)
        assert isinstance(slab, Slab)
        slab = Slab(10, p, 3)
        assert isinstance(slab, Slab)

        # make a full structure and check that the reflectivity calc works
        air = SLD(0)
        sio2 = MaterialSLD("SiO2", density=2.2)
        si = MaterialSLD("Si", density=2.33)
        s = air | sio2(10, 3) | si(0, 3)
        s.reflectivity(np.linspace(0.005, 0.3, 100))

        p = s.parameters
        assert len(list(flatten(p))) == 5 + 4 + 4

    def test_repr_slab(self):
        p = SLD(5 + 1j)
        t = p(10.5, 3.0)
        t.vfsolv = 0.1
        t.interfaces = Linear()
        q = eval(repr(t))
        assert isinstance(q, Slab)
        assert_equal(float(q.thick), 10.5)
        assert_equal(float(t.sld.real), 5)
        assert_equal(float(t.sld.imag), 1)
        assert_equal(float(q.vfsolv), 0.1)
        assert isinstance(q.interfaces, Linear)

        t.name = "pop"
        q = eval(repr(t))
        assert t.name == q.name

    def test_repr_structure(self):
        p = SLD(5 + 1j)
        t = p(10.5, 3.0)
        t.vfsolv = 0.1
        s = t | t
        q = eval(repr(s))
        assert isinstance(q, Structure)
        assert_equal(float(q[0].thick), 10.5)
        assert_equal(float(q[1].sld.real), 5)
        assert_equal(float(q[1].sld.imag), 1)

        s.name = "pop"
        q = eval(repr(s))
        assert hasattr(q, "_solvent")
        assert s.name == q.name

    def test_sld(self):
        p = SLD(5 + 1j, name="pop")
        assert_equal(float(p.real), 5)
        assert_equal(float(p.imag), 1)

        # test that we can cast to complex
        assert_equal(complex(p), 5 + 1j)

        p = SLD(5)
        assert_equal(float(p.real), 5)
        q = Parameter(5)
        r = Parameter(1)
        p = SLD([q, r])
        assert_equal(float(p.real), 5)
        assert_equal(float(p.imag), 1)

        # use SLD to make a Slab
        thickness = Parameter(100)
        roughness = Parameter(3.0)
        vfsolv = Parameter(0.2)
        s = p(thickness, roughness)
        assert_equal(s.thick.value, thickness.value)
        assert_equal(s.rough.value, roughness.value)
        assert_equal(s.vfsolv.value, 0)

        s = p(thickness, roughness, vfsolv)
        assert_equal(s.thick.value, thickness.value)
        assert_equal(s.rough.value, roughness.value)
        assert_equal(s.vfsolv.value, vfsolv.value)

        # check that we can construct SLDs from a constrained par
        deut_par = Parameter(6.36)
        h2o_solvent = SLD(-0.56)

        ms_val = 0.6 * deut_par + 0.4 * h2o_solvent.real
        mixed_solvent = SLD(ms_val)
        assert isinstance(mixed_solvent.real, _BinaryOp)
        sld = complex(mixed_solvent)
        assert_allclose(sld.real, 0.6 * 6.36 + 0.4 * -0.56)

        deut_par.value = 5.0
        sld = complex(mixed_solvent)
        assert_allclose(sld.real, 0.6 * 5.0 + 0.4 * -0.56)

    def test_sld_slicer(self):
        q = np.linspace(0.005, 0.2, 100)

        reflectivity = self.s.reflectivity(q)
        z, sld = self.s.sld_profile(z=np.linspace(-150, 250, 1000))
        round_trip_structure = _profile_slicer(z, sld, slice_size=0.5)
        round_trip_reflectivity = round_trip_structure.reflectivity(q)
        assert_allclose(round_trip_reflectivity, reflectivity, rtol=0.004)

    def test_slab_addition(self):
        # The slabs method for the main Structure component constructs
        # the overall slabs by concatenating Component slabs. This checks that
        # the slab concatenation is correct.

        si = SLD(2.07)
        sio2 = SLD(3.47)
        polymer = SLD(1.5)
        d2o = SLD(6.36)
        d2o_layer = d2o(0, 3)
        polymer_layer = polymer(20, 3)
        a = Spline(400, [4, 5.9], [0.2, 0.4], zgrad=True)
        film = si | sio2(10, 3) | polymer_layer | a | d2o_layer
        film.sld_profile()

        structure = si(0, 0)
        for i in range(200):
            p = SLD(i)(i, i)
            structure |= p

        structure |= d2o(0, 3)

        slabs = structure.slabs()
        assert_equal(slabs[1:-1, 0], np.arange(200))
        assert_equal(slabs[1:-1, 1], np.arange(200))
        assert_equal(slabs[1:-1, 3], np.arange(200))
        assert_equal(slabs[-1, 1], 6.36)
        assert_equal(slabs[0, 1], 2.07)
        assert_equal(len(slabs), 202)

    def test_component_mul(self):
        si = SLD(2.07)
        sio2 = SLD(3.47)
        polymer = SLD(1.5)
        d2o = SLD(6.36)

        s = si | sio2(10, 3) | polymer(100, 3) * 5 | d2o(0, 3)
        slabs = s.slabs()
        assert_almost_equal(np.sum(slabs[:, 0]), 510)

        s = polymer(100, 3) * 5
        assert isinstance(s, Structure)
        slabs = s.slabs()
        assert_almost_equal(np.sum(slabs[:, 0]), 500)

        # multiplying a structure should work because it extends UserList
        s = sio2(10, 3) | polymer(100, 5) * 5
        q = s * 5
        assert isinstance(q, Structure)
        assert len(q) == 30
        slabs = q.slabs()
        assert_almost_equal(np.sum(slabs[:, 0]), 510 * 5)

        # test multiplying a Stack
        stk = Stack()
        stk.append(sio2(10, 3))
        stk.append(polymer(100, 3))
        stk.repeats.value = 5

        q = stk * 3
        assert isinstance(stk * 3, Structure)
        for c in q:
            assert isinstance(c, Stack)
            assert_equal(len(c), 2)

        s = si | stk * 3 | d2o(0, 3)
        assert_equal(len(s), 5)
        slabs = s.slabs()
        assert_almost_equal(np.sum(slabs[:, 0]), 110 * 3 * 5)

    def test_contraction(self):
        q = np.linspace(0.005, 0.2, 100)

        self.s.contract = 0
        reflectivity = self.s.reflectivity(q)

        self.s.contract = 0.5
        assert_allclose(self.s.reflectivity(q), reflectivity)

        z, sld = self.s.sld_profile(z=np.linspace(-150, 250, 1000))
        slice_structure = _profile_slicer(z, sld, slice_size=0.5)
        slice_structure.contract = 0.02
        slice_reflectivity = slice_structure.reflectivity(q)
        assert_allclose(slice_reflectivity, reflectivity, rtol=5e-3)

        # test cythonized contract_by_area code
        try:
            from refnx.reflect._creflect import _contract_by_area as ca2
            from refnx.reflect._reflect import _contract_by_area as ca

            slabs = slice_structure.slabs()
            assert_almost_equal(ca2(slabs, 2), ca(slabs, 2))
        except ImportError:
            pass

    def test_stack(self):
        stk = Stack()
        slabs = stk.slabs(None)
        assert slabs is None

        si = SLD(2.07)
        sio2 = SLD(3.47)
        polymer = SLD(1.0)
        d2o = SLD(6.36)

        # check some initial stack properties
        stk.append(sio2(55, 4))
        slabs = stk.slabs(None)
        assert slabs.shape == (1, 5)
        assert_equal(np.sum(slabs[:, 0]), 55)
        assert_equal(slabs[0, 1], 3.47)
        stk.repeats.value = 3.2
        slabs = stk.slabs(None)
        assert slabs.shape == (3, 5)
        assert_equal(np.sum(slabs[:, 0]), 165)

        # ior a Stack and a Component
        stk |= polymer(110, 3.5)
        assert_equal(len(stk), 2)
        assert isinstance(stk, Stack)
        assert_almost_equal(stk.repeats, 3.2)
        slabs = stk.slabs()
        assert slabs.shape == (6, 5)
        assert_equal(np.sum(slabs[:, 0]), 495)

        # place a stack into a structure
        s = si | d2o(10, 3) | stk | d2o
        assert isinstance(s, Structure)
        slabs = s.slabs()
        assert_equal(slabs[:, 0], [0, 10, 55, 110, 55, 110, 55, 110, 0])
        assert_equal(
            slabs[:, 1], [2.07, 6.36, 3.47, 1.0, 3.47, 1.0, 3.47, 1.0, 6.36]
        )
        assert_equal(slabs[:, 3], [0, 3, 4, 3.5, 4, 3.5, 4, 3.5, 0])

        # what are the interfaces of the Stack
        assert_equal(len(stk.interfaces), len(stk.slabs()))
        assert_equal(len(list(flatten(s.interfaces))), len(s.slabs()))

        # ior a Structure and a Stack
        s = Structure(components=[si(), d2o(10, 3)])
        s |= stk
        s |= d2o
        assert isinstance(s, Structure)

        assert_equal(s.slabs()[:, 0], [0, 10, 55, 110, 55, 110, 55, 110, 0])
        assert_equal(
            s.slabs()[:, 1],
            [2.07, 6.36, 3.47, 1.0, 3.47, 1.0, 3.47, 1.0, 6.36],
        )

        q = repr(s)
        r = eval(q)
        assert_equal(r.slabs()[:, 0], [0, 10, 55, 110, 55, 110, 55, 110, 0])
        assert_equal(
            r.slabs()[:, 1],
            [2.07, 6.36, 3.47, 1.0, 3.47, 1.0, 3.47, 1.0, 6.36],
        )

        s |= stk
        assert isinstance(s.components[-1], Stack)
        import pytest

        with pytest.raises(ValueError):
            s.slabs()
