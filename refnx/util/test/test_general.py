from __future__ import division

import unittest
import refnx.util.general as general
import numpy as np
import os
from numpy.testing import assert_equal

class TestGeneral(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(__file__)
        self.path = path

    def test_q(self):
        q = general.q(1., 2.)
        assert_almost_equal(q, 0.1096567037)
