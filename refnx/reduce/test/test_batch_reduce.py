import unittest
import numpy as np
import os.path
from refnx.reduce import BatchReducer


class TestReduce(unittest.TestCase):
    def setUp(self):
        path = os.path.dirname(__file__)
        self.path = path

        return 0

    def test_batch_reduce(self):
        filename = os.path.join(self.path, "test_batch_reduction.xls")
        b = BatchReducer(filename, 2, self.path)

        b.reduce(show=False)


if __name__ == '__main__':
    unittest.main()
