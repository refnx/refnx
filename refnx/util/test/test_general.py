from __future__ import division

import unittest
import refnx.util.general as general
import refnx
import numpy as np
import os
from numpy.testing import assert_almost_equal


def test_version():
    # check that we can retrieve a version string
    version = refnx.__version__


class TestGeneral(unittest.TestCase):
    def setUp(self):
        path = os.path.dirname(__file__)
        self.path = path

    def test_q(self):
        q = general.q(1., 2.)
        assert_almost_equal(q, 0.1096567037)

    def test_q2(self):
        qx, qy, qz = general.q2(1., 2., 0., 2.)
        assert_almost_equal(qz, 0.1096567037)

    def test_wavelength_velocity(self):
        speed = general.wavelength_velocity(20.)
        assert_almost_equal(speed, 197.8017006541796, 5)

    def test_wavelength(self):
        wavelength = general.wavelength(0.1096567037, 1.)
        assert_almost_equal(wavelength, 2.)

    def test_angle(self):
        angle = general.angle(0.1096567037, 2.)
        assert_almost_equal(angle, 1.)


if __name__ == '__main__':
    unittest.main()
