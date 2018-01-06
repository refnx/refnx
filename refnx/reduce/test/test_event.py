import os
import numpy as np
from numpy.testing import assert_equal
import refnx.reduce.event as event
from refnx.reduce.event import events
from refnx.reduce import PlatypusNexus

try:
    import refnx.reduce._cevent as _cevent
except ImportError:
    HAVE_CEVENTS = False
else:
    HAVE_CEVENTS = True


class TestEvent(object):

    @classmethod
    def setup_class(cls):
        path = os.path.dirname(os.path.realpath(__file__))
        cls.event_file_path = os.path.join(path,
                                           'DAQ_2012-01-19T15-45-52',
                                           'DATASET_0',
                                           'EOS.bin')

        with open(cls.event_file_path, 'rb') as f:
            if HAVE_CEVENTS:
                event_list, fpos = _cevent._cevents(f)
            else:
                event_list, fpos = event._events(f)

        cls.event_list = event_list
        cls.fpos = fpos
        cls.f, cls.t, cls.y, cls.x = event_list

    def setup_method(self):
        path = os.path.dirname(os.path.realpath(__file__))
        self.path = path

    def test_events_smoke(self):
        # check that the event.events function works
        events(self.event_file_path)

    def test_num_events(self):
        assert_equal(1056618, self.x.size)

    def test_max_frames(self):
        # test reading only a certain number of frames
        # also use this test to compare pure python read of events
        with open(self.event_file_path, 'rb') as g:
            event_list, fpos = event._events(g, max_frames=10)

        f, t, y, x = event_list
        max_f = np.max(f)
        assert_equal(9, max_f)

        if HAVE_CEVENTS:
            with open(self.event_file_path, 'rb') as g:
                event_list, fpos = _cevent._cevents(g, max_frames=10)
                cyf, cyt, cyy, cyx = event_list

            max_f = np.max(cyf)
            assert_equal(9, max_f)

            assert_equal(cyf, f)
            assert_equal(cyt, t)
            assert_equal(cyy, y)
            assert_equal(cyx, x)

    def test_event_same_as_detector(self):
        # the detector file should be the same as the event file
        orig_file = PlatypusNexus(os.path.join(self.path,
                                               'PLP0011613.nx.hdf'))
        orig_det = orig_file.cat.detector
        frames = [np.arange(0, 23999)]
        event_det, fb = event.process_event_stream(self.event_list,
                                                   frames,
                                                   orig_file.cat.t_bins,
                                                   orig_file.cat.y_bins,
                                                   orig_file.cat.x_bins)
        assert_equal(event_det, orig_det)

        # PlatypusNexus.process_event_stream should be the same as well
        det, fc, bm = orig_file.process_event_stream(frame_bins=[])
        assert_equal(det, orig_det)

    def test_open_with_path(self):
        # give the event reader a file path
        event_list, fpos = event._events(self.event_file_path, max_frames=10)
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

        frames = event.framebins_to_frames(np.linspace(0, 24000, 7))
        detector, fbins = event.process_event_stream(self.event_list,
                                                     frames,
                                                     t_bins,
                                                     y_bins,
                                                     x_bins)
        assert_equal(detector[0, 382, 97, 0], 70)
        assert_equal(detector[1, 383, 97, 0], 64)
        assert_equal(detector[4, 377, 98, 0], 57)

        x_bins = np.array([210.5, -210.5])
        y_bins = np.linspace(110.5, -110.5, 222)
        t_bins = np.linspace(0, 50000, 1001)
        frames = [[6, 10]]
        detector, fbins = event.process_event_stream(self.event_list,
                                                     frames,
                                                     t_bins,
                                                     y_bins,
                                                     x_bins)
        assert_equal(np.sum(detector.ravel()), 167)

        # # now see what happens if we go too far with the frame_bins
        # frames = event.framebins_to_frames([0, 24000, 30000])
        # detector, fbins = event.process_event_stream(self.event_list,
        #                                              frames,
        #                                              t_bins,
        #                                              y_bins,
        #                                              x_bins)
        # assert_equal(np.size(detector, 0), 1)
