import unittest
import refnx.reduce.platypusnexus as plp
import numpy as np
import os
from numpy.testing import assert_almost_equal


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


if __name__ == '__main__':
    unittest.main()
