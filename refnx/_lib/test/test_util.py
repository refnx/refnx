import unittest
import os.path
from numpy.testing import assert_equal

from refnx._lib import flatten

path = os.path.dirname(os.path.abspath(__file__))


class TestUtil(unittest.TestCase):

    def setUp(self):
        pass

    def test_flatten(self):
        l = [1, 2, [3, 4, 5], 6, 7]
        t = list(flatten(l))
        assert_equal(t, [1, 2, 3, 4, 5, 6, 7])


if __name__ == '__main__':
    unittest.main()
