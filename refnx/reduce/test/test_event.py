import unittest
import refnx.reduce.event as event
import numpy as np
import os
from numpy.testing import assert_equal

class TestEvent(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(__file__)
        self.path = path
        self.event_file_path = os.path.join(path,
                                       'DAQ_2012-01-19T15-45-52',
                                       'DATASET_0',
                                       'EOS.bin')

        with open(self.event_file_path, 'rb') as f:
            event_list, fpos = event.events(f)

        self.event_list = event_list
        self.fpos = fpos
        self.f, self.t, self.y, self.x = event_list
        
    def test_num_events(self):
        assert_equal(1056618, self.x.size)

    def test_max_frames(self):
        # test reading only a certain number of frames
        with open(self.event_file_path, 'rb') as f:
            event_list, fpos = event.events(f, max_frames=10)
        f, t, y, x = event_list
        max_f = np.max(f)
        assert_equal(9, max_f)

    def test_values(self):
        # We know the values of all the events in the file from another program
        # test that a set of random events are correct.
        assert_equal(self.t[0], 47350)
        assert_equal(self.x[0], 18)
        assert_equal(self.y[0], 96)
        assert_equal(self.f[0], 5)

        assert_equal(self.t[-1], 31343)
        assert_equal(self.x[-1], 4)
        assert_equal(self.y[-1], 13)
        assert_equal(self.f[-1], 23998)

    def test_process_event_stream(self):
        x_bins = np.array([60.5, -60.5])
        y_bins = np.linspace(110.5, -110.5, 222)
        t_bins = np.linspace(0, 50000, 1001)
        f_bins = np.linspace(0, 24000, 7)
        detector, fbins = event.process_event_stream(self.event_list,
                                                     f_bins,
                                                     t_bins,
                                                     y_bins,
                                                     x_bins)
        assert_equal(detector[0, 382, 97, 0], 70)
        assert_equal(detector[1, 383, 97, 0], 64)
        assert_equal(detector[4, 377, 98, 0], 57)


if __name__ == '__main__':
    unittest.main()