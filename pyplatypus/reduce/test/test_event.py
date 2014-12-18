import unittest
import pyplatypus.reduce.event as event
import numpy as np
import os
from numpy.testing import assert_equal

class TestEvent(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(__file__)
        self.path = path
        event_file_path = os.path.join(path, 'DAQ_2012-01-19T15-45-52',
                                       'DATASET_0', 'EOS.bin')

        with open(event_file_path, 'rb') as f:
            event_list, fpos = event.events(f)

        self.event_list = event_list
        self.fpos = fpos
        self.x, self.y, self.t, self.f = event_list
        
    def test_num_events(self):
        assert_equal(1056618, self.x.size)

    def test_values(self):
        assert_equal(self.t[0], 47350)
        assert_equal(self.x[0], 18)
        assert_equal(self.y[0], 96)
        assert_equal(self.f[0], 5)

        assert_equal(self.t[-1], 31343)
        assert_equal(self.x[-1], 4)
        assert_equal(self.y[-1], 13)
        assert_equal(self.f[-1], 23998)

        
if __name__ == '__main__':
    unittest.main()