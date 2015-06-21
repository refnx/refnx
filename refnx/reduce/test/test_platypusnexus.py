import unittest
import refnx.reduce.platypusnexus as plp
import numpy as np
import os
from numpy.testing import assert_almost_equal
from refnx.reduce.peak_utils import gauss

class TestPlatypusNexus(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(__file__)
        self.path = path

        self.f113 = plp.PlatypusNexus(os.path.join(self.path,
                                                   'PLP0011613.nx.hdf'))
        self.f641 = plp.PlatypusNexus(os.path.join(self.path,
                                                   'PLP0011641.nx.hdf'))
        return 0

    def test_deflection(self):
        y = plp.deflection(20, 20000, 0.14365919135097724)
        assert_almost_equal(y, 0)

    def test_rebinning(self):
        bins = plp.calculate_wavelength_bins(2., 18, 2.)
        assert_almost_equal(bins[0], 1.98)
        assert_almost_equal(bins[-1], 18.18)

    def test_chod(self):
        flight_length = self.f113.chod()
        assert_almost_equal(flight_length[0], 7141.413818359375)
        assert_almost_equal(flight_length[1], 808)

        flight_length = self.f641.chod(omega=1.8, two_theta=3.6)
        assert_almost_equal(flight_length[0], 7146.3567785516016)
        assert_almost_equal(flight_length[1], 808)

    def test_phase_angle(self):
        # TODO. Add more tests where the phase_angle isn't zero.
        phase_angle, master_opening = self.f641.phase_angle()
        assert_almost_equal(phase_angle, 0)
        assert_almost_equal(master_opening, 1.04719755)

    def test_background_subtract_line(self):
        # checked each step of the background subtraction with IGOR
        # so this test background correction should be correct.

        # create some test data
        xvals = np.linspace(-10, 10, 201)
        yvals = np.ceil(gauss(xvals, 0, 100, 0, 1) + 2 * xvals + 30)

        # add some reproducible random noise
        np.random.seed(1)
        yvals += np.sqrt(yvals) * np.random.randn(yvals.size)
        yvals_sd = np.sqrt(yvals)

        mask = np.zeros(201, np.bool)
        mask[30:70] = True
        mask[130:160] = True

        profile, profile_sd = plp.background_subtract_line(yvals,
                                                           yvals_sd,
                                                           mask)

        verified_data = np.load(os.path.join(self.path,
                                             'background_subtract.npy'))

        assert_almost_equal(verified_data, np.c_[profile, profile_sd])


if __name__ == '__main__':
    unittest.main()
