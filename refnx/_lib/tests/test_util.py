from multiprocessing import Pool
from multiprocessing.pool import Pool as PWL
from importlib import resources


from pytest import raises as assert_raises
from numpy.testing import assert_equal

import refnx
import refnx.analysis
from refnx._lib.util import flatten, unique, MapWrapper, possibly_open_file
from refnx._lib._cutil import c_flatten

import numpy as np


class TestUtil:
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

    def test_possibly_open_file(self):
        pth = resources.files(refnx.analysis)
        datadir = pth / "tests"

        with possibly_open_file(datadir / "e361r.txt", "r") as f:
            assert hasattr(f, "read")


class TestMapWrapper:
    def setup_method(self):
        self.input = np.arange(10.0)
        self.output = np.sin(self.input)

    def test_serial(self):
        p = MapWrapper(1)
        assert p._mapfunc is map
        assert p.pool is None
        assert p._own_pool is False
        out = list(p.map(np.sin, self.input))
        assert_equal(out, self.output)

    def test_parallel(self):
        with MapWrapper(2) as p:
            out = p.map(np.sin, self.input)
            assert_equal(list(out), self.output)

            assert p._own_pool is True
            assert isinstance(p.pool, PWL)
            assert p._mapfunc is not None

        # the context manager should've closed the internal pool
        # check that it has by asking it to calculate again.
        with assert_raises(Exception) as excinfo:
            p.map(np.sin, self.input)

        # on py27 an AssertionError is raised, on >py27 it's a ValueError
        err_type = excinfo.type
        assert (err_type is ValueError) or (err_type is AssertionError)

        # can also set a MapWrapper up with a Pool instance
        try:
            p = Pool(2)
            q = MapWrapper(p.map)

            assert q._own_pool is False
            q.close()

            # closing the MapWrapper shouldn't close the internal pool
            # because it didn't create it
            out = p.map(np.sin, self.input)
            assert_equal(out, self.output)
        finally:
            p.close()
