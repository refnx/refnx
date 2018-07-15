from numpy.testing import assert_equal

from refnx._lib.util import flatten, unique
from refnx._lib._cutil import c_flatten

import numpy as np


class TestUtil(object):

    def setup_method(self):
        pass

    def test_flatten(self):
        test_list = [1, 2, [3, 4, 5], 6, 7]
        t = list(flatten(test_list))
        assert_equal(t, [1, 2, 3, 4, 5, 6, 7])

    def test_c_flatten(self):
        test_list = [1, 2, [3, 4, 5], 6, 7]
        t = list(c_flatten(test_list))
        assert_equal(t, [1, 2, 3, 4, 5, 6, 7])

    def test_unique(self):
        ints = [int(val) for val in np.random.randint(0, 100, size=10000)]
        num_unique = np.unique(ints).size
        num_unique2 = len(list(unique(ints)))
        assert_equal(num_unique2, num_unique)
