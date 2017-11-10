from numpy.testing import assert_equal

from refnx._lib import flatten


class TestUtil(object):

    def setup_method(self):
        pass

    def test_flatten(self):
        test_list = [1, 2, [3, 4, 5], 6, 7]
        t = list(flatten(test_list))
        assert_equal(t, [1, 2, 3, 4, 5, 6, 7])
