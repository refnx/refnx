import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from refnx.reflect import SLD, Slab, Structure, Spline


class TestReflect(object):

    def setup_method(self):
        self.left = SLD(1.5)(10, 3)
        self.right = SLD(2.5)(10, 3)
        self.solvent = SLD(10)(0, 3)

    def test_spline_smoke(self):
        # smoke test to make Spline at least gives us something
        a = Spline(100, [2],
                   [0.5], self.left, self.right, self.solvent, zgrad=False,
                   microslab_max_thickness=2)
        b = a.slabs
        assert_equal(b[:, 2], 0)
        assert_equal(b[:, 3], 0.5)

        # microslabs are assessed in the middle of the slab
        assert_equal(b[0, 1], a(0.5 * b[0, 0]))

        # with the ends turned off the profile should be a straight line
        assert_equal(a(50), 2.0)

    def test_left_right_influence(self):
        # make sure that if the left and right components change, so does the
        # spline
        a = Spline(100, [2],
                   [0.5], self.left, self.right, self.solvent, zgrad=False,
                   microslab_max_thickness=2)

        # change the SLD of the left component, spline should respond
        self.left.sld.real.value = 2.
        assert_almost_equal(a(0), 2)

        # check that the spline responds if it's a vfsolve that changes
        self.left.vfsolv.value = 0.5
        assert_almost_equal(Structure.overall_sld(self.left.slabs,
                                                  self.solvent.slabs)[0, 1],
                            6.)
        assert_almost_equal(a(0), 6.)

        # check that the right side responds.
        self.right.sld.real.value = 5.
        assert_almost_equal(a(100), 5.)

        # the spline should respond if the knot SLD's are changed
        a.vs[0].value = 3.
        assert_almost_equal(a(50), 3.)

        # spline responds if the interval knot spacing is changed
        a.dz[0].value = 0.9
        assert_almost_equal(a(90), 3.)
