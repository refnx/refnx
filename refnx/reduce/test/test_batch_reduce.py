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
        b = BatchReducer(filename, 2, self.path)

        b.reduce(show=False)

    def test_batch_reduce_ipython(self):
        filename = os.path.join(self.path, "test_batch_reduction.xls")

        refnx.reduce.batchreduction._have_ipython = False
        b = BatchReducer(filename, 2, self.path)
        b.reduce()

        refnx.reduce.batchreduction._have_ipython = True
        b = BatchReducer(filename, 2, self.path)
        b.reduce()

if __name__ == '__main__':
    unittest.main()
