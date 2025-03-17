from pathlib import Path
import numpy as np
from numpy.testing import assert_almost_equal, assert_, assert_equal
import refnx.util.nsplice as nsplice

__author__ = "anz"


class TestNSplice:
    def setup_method(self):
        self.path = Path(__file__).absolute().parent

        fname0 = self.path / "PLP0000708.dat"
        self.qv0, self.rv0, self.drv0 = np.loadtxt(
            fname0, usecols=(0, 1, 2), unpack=True, skiprows=1
        )

        fname1 = self.path / "PLP0000709.dat"
        self.qv1, self.rv1, self.drv1 = np.loadtxt(
            fname1, usecols=(0, 1, 2), unpack=True, skiprows=1
        )
        self.scale = 0.1697804743373886
        self.dscale = 0.0018856527316943896

        self.x0 = np.arange(10.0)
        self.x1 = np.arange(10.0) + 5.5
        self.y0 = self.x0 * 2
        self.y1 = self.x1 * 4
        self.dy0 = np.ones_like(self.y0)
        self.dy1 = np.ones_like(self.y1)

    def test_nsplice(self):
        # test splicing of two datasets
        scale, dscale, overlap = nsplice.get_scaling_in_overlap(
            self.qv0, self.rv0, self.drv0, self.qv1, self.rv1, self.drv1
        )
        assert_almost_equal(scale, self.scale)
        assert_almost_equal(dscale, self.dscale)

    def test_nsplice_nooverlap(self):
        # test splicing of two datasets if there's no overlap
        # the scale factor should be np.nan
        qv0 = self.qv0[0:10]
        scale, dscale, overlap = nsplice.get_scaling_in_overlap(
            qv0, self.rv0, self.drv0, self.qv1, self.rv1, self.drv1
        )
        assert_(not np.isfinite(scale))
        assert_(not np.isfinite(dscale))
        assert_equal(np.size(overlap, 0), 0)

    def test_nsplice_notsorted(self):
        # test splicing of two datasets if there's no sorting
        # of the data
        vec0 = np.arange(np.size(self.qv0))
        np.random.shuffle(vec0)
        vec1 = np.arange(np.size(self.qv1))
        np.random.shuffle(vec1)

        scale, dscale, overlap = nsplice.get_scaling_in_overlap(
            self.qv0[vec0],
            self.rv0[vec0],
            self.drv0[vec0],
            self.qv1[vec1],
            self.rv1[vec1],
            self.drv1[vec1],
        )

        assert_almost_equal(scale, self.scale)
        assert_almost_equal(dscale, self.dscale)

    def test_nsplice_overlap_points(self):
        scale, dscale, overlap = nsplice.get_scaling_in_overlap(
            self.x0, self.y0, self.dy0, self.x1, self.y1, self.dy1
        )

        assert_almost_equal(scale, 0.5)
        assert_equal(np.count_nonzero(overlap), 4)
