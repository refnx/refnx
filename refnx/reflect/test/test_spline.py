import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_almost_equal,
                           assert_)
from refnx.reflect import SLD, Slab, Structure, Spline
from refnx.analysis import Parameter, Interval, Parameters
from refnx._lib import flatten


class TestReflect(object):

    def setup_method(self):
        self.left = SLD(1.5)(10, 3)
        self.right = SLD(2.5)(10, 3)
        self.solvent = SLD(10)

    def test_spline_smoke(self):
        # smoke test to make Spline at least gives us something
        a = Spline(100, [2],
                   [0.5], zgrad=False, microslab_max_thickness=1)

        s = self.left | a | self.right | self.solvent
        b = a.slabs(s)
        assert_equal(b[:, 2], 0)

        # microslabs are assessed in the middle of the slab
        assert_equal(b[0, 1], a(0.5 * b[0, 0], s))

        # with the ends turned off the profile should be a straight line
        assert_equal(a(50, s), 2.0)

        # construct a structure
        a = Spline(100, [2., 3., 4.],
                   [0.25] * 3, zgrad=False, microslab_max_thickness=1)

        # s.solvent = None
        s = self.left | a | self.right | self.solvent
        # calculate an SLD profile
        s.sld_profile()
        # ask for the parameters
        for p in flatten(s.parameters):
            assert_(isinstance(p, Parameter))

        # s.solvent is not None
        s.solvent = self.solvent
        # calculate an SLD profile
        s.sld_profile()

    def test_spline_solvation(self):
        a = Spline(100, [2],
                   [0.5], zgrad=False, microslab_max_thickness=1)

        front = SLD(0.1)
        air = SLD(0.0)
        s = front | self.left | a | self.right | self.solvent

        # assign a solvent
        s.solvent = air
        self.left.vfsolv.value = 0.5
        self.right.vfsolv.value = 0.5
        assert_equal(s.slabs()[1, 1], 0.75)
        assert_equal(s.slabs()[-2, 1], 1.25)

        assert_almost_equal(a(0, s), 0.75)
        assert_almost_equal(a(100, s), 1.25)

        # remove solvent, should be solvated by backing medium
        s.solvent = None
        assert_equal(s.slabs()[1, 1], 5.75)
        assert_equal(s.slabs()[-2, 1], 6.25)
        assert_almost_equal(a(0, s), 5.75)
        assert_almost_equal(a(100, s), 6.25)

        # reverse structure, should be solvated by fronting medium
        s.reverse_structure = True
        assert_equal(s.slabs()[1, 1], 1.3)
        assert_equal(s.slabs()[-2, 1], 0.8)
        # note that a(0, s) end becomes the end when the structure is reversed
        assert_almost_equal(a(0, s), 0.8)
        assert_almost_equal(a(100, s), 1.3)

    def test_left_right_influence(self):
        # make sure that if the left and right components change, so does the
        # spline
        a = Spline(100, [2],
                   [0.5], zgrad=False, microslab_max_thickness=1)

        s = self.left | a | self.right | self.solvent

        # change the SLD of the left component, spline should respond
        self.left.sld.real.value = 2.
        assert_almost_equal(a(0, s), 2)

        # check that the spline responds if it's a vfsolv that changes
        self.left.vfsolv.value = 0.5
        assert_almost_equal(Structure.overall_sld(self.left.slabs(),
                                                  self.solvent)[0, 1],
                            6.)
        assert_almost_equal(a(0, s), 6.)

        # check that the right side responds.
        self.right.sld.real.value = 5.
        assert_almost_equal(a(100, s), 5.)

        # the spline should respond if the knot SLD's are changed
        a.vs[0].value = 3.
        assert_almost_equal(a(50, s), 3.)

        # spline responds if the interval knot spacing is changed
        a.dz[0].value = 0.9
        assert_almost_equal(a(90, s), 3.)

    def test_repr(self):
        # make sure that if the left and right components change, so does the
        # spline
        a = Spline(100, [2, 3, 4],
                   [0.1, 0.2, 0.3], zgrad=False, microslab_max_thickness=1)

        s = self.left | a | self.right | self.solvent

        # should be able to repr the whole lot
        q = repr(s)
        r = eval(q)
        assert_equal(r.slabs(), s.slabs())

    def test_spline_repeat(self):
        # can't have two splines in a row.
        a = Spline(100, [2],
                   [0.5], zgrad=False, microslab_max_thickness=1)

        s = self.left | a | a | self.right | self.solvent

        from pytest import raises
        with raises(ValueError):
            s.slabs()
