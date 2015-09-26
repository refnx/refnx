import unittest
import refnx.reduce.platypusnexus as plp
import numpy as np
import os
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
                           assert_array_less)

class TestPlatypusNexus(unittest.TestCase):

    def setUp(self):
    path = os.path.dirname(__file__)
    self.path = path

    return 0