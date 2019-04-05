import pickle

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_allclose)

from refnx._lib import flatten
from refnx.reflect import (SLD, Structure, Spline, Slab, Stack)
from refnx.reflect.structure import _profile_slicer
from refnx.analysis import Parameter, Interval, Parameters


class TestStructure(object):

    def setup_method(self):
        self.air = SLD(0, name='air')
        self.sio2 = SLD(3.47, name='sio2')
        self.d2o = SLD(6.36, name='d2o')
        self.h2o = SLD(-0.56, name='h2o')
        self.s = self.air | self.sio2(100, 5) | self.d2o(0, 4)

    def test_structure_construction(self):
        # structures are constructed by or-ing slabs
        # test that the slab representation is correct
        assert_equal(self.s.slabs(), np.array([[0, 0, 0, 0, 0],
                                              [100, 3.47, 0, 5, 0],
                                              [0, 6.36, 0, 4, 0]]))

        self.s[1] = SLD(3.47 + 1j, name='sio2')(100, 5)
        self.s[1].vfsolv.value = 0.9

        oldpars = len(list(flatten(self.s.parameters)))

        # slabs have solvent penetration
        self.s.solvent = SLD(5 + 1.2j)
        sld = 5 * 0.9 + 0.1 * 3.47
        sldi = 1 * 0.1 + 0.9 * 1.2
        assert_almost_equal(self.s.slabs(), np.array([[0, 0, 0, 0, 0],
                                                      [100, sld, sldi, 5, 0.9],
                                                      [0, 6.36, 0, 4, 0]]))

        # when the structure._solvent is not None, but an SLD object, then
        # it's number of parameters should increase by 2.
        newpars = len(list(flatten(self.s.parameters)))
        assert_equal(newpars, oldpars + 2)

        # by default solvation is done by backing medium
        self.s.solvent = None
        sld = 6.36 * 0.9 + 0.1 * 3.47
        sldi = 1 * 0.1
        assert_almost_equal(self.s.slabs(), np.array([[0, 0, 0, 0, 0],
                                                     [100, sld, sldi, 5, 0.9],
                                                     [0, 6.36, 0, 4, 0]]))

        # by default solvation is done by backing medium, except when structure
        # is reversed
        self.s.reverse_structure = True
        sld = 0 * 0.9 + 0.1 * 3.47
        sldi = 0 * 0.9 + 1 * 0.1
        assert_almost_equal(self.s.slabs(), np.array([[0, 6.36, 0, 0, 0],
                                                      [100, sld, sldi, 4, 0.9],
                                                      [0, 0, 0, 5, 0]]))

    def test_pickle(self):
        # need to be able to pickle and unpickle structure
        pkl = pickle.dumps(self.s)
        unpkl = pickle.loads(pkl)
        assert_(isinstance(unpkl, Structure))
        for param in unpkl.parameters.flattened():
            assert_(isinstance(param, Parameter))
        assert(hasattr(unpkl, '_solvent'))

    def test_sld_profile(self):
        # check that it runs
        z, sld_profile = self.s.sld_profile()

        z, sld_profile = self.s.sld_profile(np.linspace(-100, 100, 100))
        assert_equal(min(z), -100)
        assert_equal(max(z), 100)

    def test_reflectivity(self):
        q = np.linspace(0.005, 0.3, 200)
        self.s.reflectivity(q)

    def test_repr_sld(self):
        p = SLD(5 + 1j, name='pop')
        assert_equal(float(p.real), 5)
        assert_equal(float(p.imag), 1)
        print(repr(p))
        q = eval(repr(p))
        assert_equal(float(q.real), 5)
        assert_equal(float(q.imag), 1)

    def test_repr_slab(self):
        p = SLD(5 + 1j)
        t = p(10.5, 3.)
        t.vfsolv = 0.1
        q = eval(repr(t))
        assert(isinstance(q, Slab))
        assert_equal(float(q.thick), 10.5)
        assert_equal(float(t.sld.real), 5)
        assert_equal(float(t.sld.imag), 1)
        assert_equal(float(q.vfsolv), 0.1)

        t.name = 'pop'
        q = eval(repr(t))
        assert(t.name == q.name)

    def test_repr_structure(self):
        p = SLD(5 + 1j)
        t = p(10.5, 3.)
        t.vfsolv = 0.1
        s = t | t
        q = eval(repr(s))
        assert(isinstance(q, Structure))
        assert_equal(float(q[0].thick), 10.5)
        assert_equal(float(q[1].sld.real), 5)
        assert_equal(float(q[1].sld.imag), 1)

        s.name = 'pop'
        q = eval(repr(s))
        assert(hasattr(q, '_solvent'))
        assert(s.name == q.name)

    def test_sld(self):
        p = SLD(5 + 1j, name='pop')
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

    def test_sld_slicer(self):
        q = np.linspace(0.005, 0.2, 100)

        reflectivity = self.s.reflectivity(q)
        z, sld = self.s.sld_profile(z=np.linspace(-150, 250, 1000))
        round_trip_structure = _profile_slicer(z, sld, slice_size=0.1)
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
        a = Spline(400, [4, 5.9],
                   [0.2, .4], zgrad=True)
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

    def test_contraction(self):
        q = np.linspace(0.005, 0.2, 100)

        self.s.contract = 0
        reflectivity = self.s.reflectivity(q)

        self.s.contract = 0.5
        assert_allclose(self.s.reflectivity(q),
                        reflectivity)

        z, sld = self.s.sld_profile(z=np.linspace(-150, 250, 1000))
        slice_structure = _profile_slicer(z, sld, slice_size=0.05)
        slice_structure.contract = 0.02
        slice_reflectivity = slice_structure.reflectivity(q)
        assert_allclose(slice_reflectivity,
                        reflectivity,
                        rtol=5e-3)

    def test_stack(self):
        stk = Stack()
        slabs = stk.slabs(None)
        assert(slabs is None)

        si = SLD(2.07)
        sio2 = SLD(3.47)
        polymer = SLD(1.0)
        d2o = SLD(6.36)

        # check some initial stack properties
        stk.append(sio2(55, 4))
        slabs = stk.slabs(None)
        assert(slabs.shape == (1, 5))
        assert_equal(np.sum(slabs[:, 0]), 55)
        assert_equal(slabs[0, 1], 3.47)
        stk.repeats.value = 3.2
        slabs = stk.slabs(None)
        assert(slabs.shape == (3, 5))
        assert_equal(np.sum(slabs[:, 0]), 165)

        # ior a Stack and a Component
        stk |= polymer(110, 3.5)
        assert_equal(len(stk), 2)
        assert(isinstance(stk, Stack))
        assert_almost_equal(stk.repeats, 3.2)
        slabs = stk.slabs()
        assert(slabs.shape == (6, 5))
        assert_equal(np.sum(slabs[:, 0]), 495)

        # place a stack into a structure
        s = si | d2o(10, 3) | stk | d2o
        assert(isinstance(s, Structure))
        slabs = s.slabs()
        assert_equal(slabs[:, 0], [0, 10, 55, 110, 55, 110, 55, 110, 0])
        assert_equal(slabs[:, 1],
                     [2.07, 6.36, 3.47, 1.0, 3.47, 1.0, 3.47, 1.0, 6.36])
        assert_equal(slabs[:, 3],
                     [0, 3, 4, 3.5, 4, 3.5, 4, 3.5, 0])

        # ior a Structure and a Stack
        s = Structure(components=[si(), d2o(10, 3)])
        s |= stk
        s |= d2o
        assert(isinstance(s, Structure))

        assert_equal(s.slabs()[:, 0], [0, 10, 55, 110, 55, 110, 55, 110, 0])
        assert_equal(s.slabs()[:, 1],
                     [2.07, 6.36, 3.47, 1.0, 3.47, 1.0, 3.47, 1.0, 6.36])

        q = repr(s)
        r = eval(q)
        assert_equal(r.slabs()[:, 0], [0, 10, 55, 110, 55, 110, 55, 110, 0])
        assert_equal(r.slabs()[:, 1],
                     [2.07, 6.36, 3.47, 1.0, 3.47, 1.0, 3.47, 1.0, 6.36])

        s |= stk
        assert(isinstance(s.components[-1], Stack))
        import pytest
        with pytest.raises(ValueError):
            s.slabs()
