import os.path
import os
import pandas
import tempfile

from numpy.testing import assert_equal, assert_
import pytest

from refnx.reduce import BatchReducer
# also get access to file-scope variables
import refnx.reduce.batchreduction


class TestReduce(object):

    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir):
        path = os.path.dirname(__file__)
        self.path = path

        self.cwd = os.getcwd()

        self.tmpdir = tmpdir.strpath
        os.chdir(self.tmpdir)
        return 0

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_batch_reduce(self):
        filename = os.path.join(self.path, "test_batch_reduction.xls")
        b = BatchReducer(filename, data_folder=self.path, verbose=False,
                         persistent=False)

        b.reduce(show=False)

    def test_batch_reduce_ipython(self):
        filename = os.path.join(self.path, "test_batch_reduction.xls")

        refnx.reduce.batchreduction._have_ipython = False
        b = BatchReducer(filename, data_folder=self.path, verbose=False,
                         persistent=False)
        b.reduce(show=False)

        refnx.reduce.batchreduction._have_ipython = True
        b = BatchReducer(filename, data_folder=self.path, verbose=False,
                         persistent=False)
        b.reduce(show=False)


class TestReductionCache(object):

    entries = [
        # row, ds, name, fname, entry
        [1, None, "Sample A", "a.dat", (1, 2, 3)],
        [2, None, "Sample B", "b.dat", (11, 12, 13)],
        [3, None, "Sample C", "c.dat", (21, 22, 23)],
        # blank row in sheet
        [5, None, "Sample A1", "a1.dat", (31, 32, 33)],
        [6, None, "Sample A2", "a2.dat", (41, 42, 43)],
        [7, None, "Sample D", "d.dat", (51, 52, 53)],
    ]

    def setup_method(self):
        self.cache = refnx.reduce.batchreduction.ReductionCache(
            persistent=False)

        # populate the cache with some fake data to test selector methods
        for line in self.entries:
            self._addentry(line)

    def _addentry(self, line, dest=None, **kwargs):
        runs = pandas.Series(dict(zip(['refl1', 'refl2', 'refl3'], line[4])))
        entry = line[:]
        entry[4] = runs
        if dest is None:
            dest = self.cache
        dest.add(*entry, **kwargs)

    def test_add(self):
        assert_equal(len(self.cache), 6)

        # test adding with update=True
        new_entry = self.entries[-1]
        new_entry[2] = 'Sample D2'
        self._addentry(self.entries[-1], update=True)
        assert_equal(len(self.cache), 6)
        assert_equal(self.cache.row(7).name, 'Sample D2')

        # test adding with update=True
        new_entry[2] = 'Sample D3'
        self._addentry(self.entries[-1], update=False)
        # the old entry is still in the list
        assert_equal(len(self.cache), 7)
        assert_(self.cache.name('Sample D2'))
        # but the new entry should be visible searching by-row
        assert_equal(self.cache.row(7).name, 'Sample D3')

    def test_run(self):
        assert_equal(self.cache.run(21).entry['refl2'], 22)
        assert_equal(self.cache.run(42).name, "Sample A2")

    def test_runs(self):
        assert_equal(len(self.cache.runs((2, 12))), 2)
        assert_(type(self.cache.runs([2])[0])
                is refnx.reduce.batchreduction.ReductionEntry)

    def test_row(self):
        from pytest import raises
        assert_equal(self.cache.row(5).name, 'Sample A1')
        with raises(KeyError):
            self.cache.row(4)
        with raises(KeyError):
            self.cache.row(8)

    def test_rows(self):
        assert_equal(len(self.cache.rows((2, 5))), 2)
        assert_equal(len(self.cache.rows((2, 4))), 1)

    def test_name(self):
        from pytest import raises

        assert_equal(self.cache.name('Sample C').fname, 'c.dat')
        with raises(KeyError):
            self.cache.name('No such sample')

    def test_name_startswith(self):
        assert_equal(len(self.cache.name_startswith('Sample A')), 3)

    def test_name_search(self):
        assert_equal(len(self.cache.name_search('^Sample A')), 3)
        assert_equal(len(self.cache.name_search('A')), 3)
        assert_equal(len(self.cache.name_search(r'A\d')), 2)
        assert_equal(len(self.cache.name_search('no such sample')), 0)

    def test_str(self):
        # smoke test the __str__ method
        assert_(str(self.cache) != '')

    def test_persistence(self, tmpdir):
        # test persistence of cache in a local file

        # presence of this file would indicate a failure of
        # the previous persistent=False
        assert_(not os.path.exists(self.cache._default_persistent_cache))

        # make a new cache that has persistence with a default name
        cache = refnx.reduce.batchreduction.ReductionCache(persistent=True)
        self._addentry(self.entries[0], dest=cache)
        assert_equal(len(cache), 1)
        assert_(cache._cache_filename() ==
                self.cache._default_persistent_cache)
        assert_(os.path.exists(cache._cache_filename()))

        # check that the persistent cache has has been dropped
        cache.drop_cache()
        assert_(not os.path.exists(cache._cache_filename()))

        # check that the cache filename can be set
        cachename = tmpdir.mkdir('test').join('redn-test.pickle').strpath

        # make a new cache that has persistence with specified filename
        cache = refnx.reduce.batchreduction.ReductionCache(
            persistent=cachename)

        assert_(cache._cache_filename() == cachename)

        self._addentry(self.entries[0], dest=cache)
        assert_equal(len(cache), 1)

        # check that the persistent cache has been written
        assert_(os.path.exists(cache._cache_filename()))
        assert_(os.path.getsize(cache._cache_filename()) > 0)

        # check that the persistent cache is loaded again
        cache2 = refnx.reduce.batchreduction.ReductionCache(
            persistent=cachename)
        assert_equal(len(cache2), 1)
