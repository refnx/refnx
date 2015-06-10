import numpy as np
import unittest
from numpy.testing import (assert_almost_equal, assert_equal)
import refnx.reduce.parabolic_motion as pm

class TestParabolicMotion(unittest.TestCase):

    def setUp(self):
        pass

    def test_y_deflection(self):
        # Launch a projectile at 45 degrees at 300 m/s.
        # It should have a flight time of 43.262894944953523
        # for which the range is 9177.445 m, at which point the
        # deflection should be 0.
        deflection = pm.y_deflection(45, 9177.4459168013527, 300.)
        assert_almost_equal(deflection, 0)


if __name__ == '__main__':
    unittest.main()