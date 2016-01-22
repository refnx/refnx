import unittest
import numpy as np
import os.path
from refnx.reduce import BatchReducer

# also get access to file-scope variables
import refnx.reduce.batchreduction


class TestReduce(unittest.TestCase):
    def setUp(self):
        path = os.path.dirname(__file__)
        self.path = path

        return 0

    def test_batch_reduce(self):
        filename = os.path.join(self.path, "test_batch_reduction.xls")
        b = BatchReducer(filename, data_folder=self.path, verbose=False)

        b.reduce(show=False)

    def test_batch_reduce_ipython(self):
        filename = os.path.join(self.path, "test_batch_reduction.xls")

        refnx.reduce.batchreduction._have_ipython = False
        b = BatchReducer(filename, data_folder=self.path, verbose=False)
        b.reduce(show=False)

        refnx.reduce.batchreduction._have_ipython = True
        b = BatchReducer(filename, data_folder=self.path, verbose=False)
        b.reduce(show=False)


class TestReductionCache(unittest.TestCase):

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

    def setUp(self):
        self.cache = refnx.reduce.batchreduction.ReductionCache()

        # populate the cache with some fake data to test selector methods
        for line in self.entries:
            self._addentry(line)

    def _addentry(self, line, **kwargs):
        runs = dict(zip(['refl1', 'refl2','refl3'], line[4]))
        entry = line[:]
        entry[4] = runs
        self.cache.add(*entry, **kwargs)

    def test_add(self):
        self.assertEqual(len(self.cache), 6)

        # test adding with update=True
        new_entry = self.entries[-1]
        new_entry[2] = 'Sample D2'
        self._addentry(self.entries[-1], update=True)
        self.assertEqual(len(self.cache), 6)
        self.assertEqual(self.cache.row(7).name, 'Sample D2')

        # test adding with update=True
        new_entry[2] = 'Sample D3'
        self._addentry(self.entries[-1], update=False)
        # the old entry is still in the list
        self.assertEqual(len(self.cache), 7)
        self.assertTrue(self.cache.name('Sample D2'))
        # but the new entry should be visible searching by-row
        self.assertEqual(self.cache.row(7).name, 'Sample D3')

    def test_run(self):
        self.assertEqual(self.cache.run(21).entry['refl2'], 22)
        self.assertEqual(self.cache.run(42).name, "Sample A2")

    def test_runs(self):
        self.assertEqual(len(self.cache.runs((2, 12))), 2)

    def test_row(self):
        self.assertEqual(self.cache.row(5).name, 'Sample A1')
        with self.assertRaises(KeyError):
            self.cache.row(4)
        with self.assertRaises(KeyError):
            self.cache.row(8)

    def test_rows(self):
        self.assertEqual(len(self.cache.rows((2, 5))), 2)
        self.assertEqual(len(self.cache.rows((2, 4))), 1)

    def test_name(self):
        self.assertEqual(self.cache.name('Sample C').fname, 'c.dat')
        with self.assertRaises(KeyError):
            self.cache.name('No such sample')

    def test_name_startswith(self):
        self.assertEqual(len(self.cache.name_startswith('Sample A')), 3)

    def test_name_search(self):
        self.assertEqual(len(self.cache.name_search('^Sample A')), 3)
        self.assertEqual(len(self.cache.name_search('A')), 3)
        self.assertEqual(len(self.cache.name_search(r'A\d')), 2)
        self.assertEqual(len(self.cache.name_search('no such sample')), 0)


if __name__ == '__main__':
    unittest.main()
