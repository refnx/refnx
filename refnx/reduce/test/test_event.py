import os
import warnings

import numpy as np
import h5py

from numpy.testing import assert_equal
import refnx.reduce.event as event
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
                                           'DAQ_2012-01-20T21-50-32',
                                           'DATASET_0',
                                           'EOS.bin')

        with open(cls.event_file_path, 'rb') as f:
            event_list, fpos = _cevent._cevents(f)

        cls.event_list = event_list
        cls.fpos = fpos
        cls.f, cls.t, cls.y, cls.x = event_list

    def setup_method(self):
        path = os.path.dirname(os.path.realpath(__file__))
        self.path = path

    def test_events_smoke(self):
        # check that the event.events function works
        event.events(self.event_file_path)

    def test_num_events(self):
        assert_equal(self.x.size, 783982)

    def test_max_frames(self):
        # test reading only a certain number of frames

        if HAVE_CEVENTS:
            with open(self.event_file_path, 'rb') as g:
                event_list, fpos = _cevent._cevents(g, max_frames=10)
                cyf, cyt, cyy, cyx = event_list

            max_f = np.max(cyf)
            assert_equal(9, max_f)

            event_list, fpos = _cevent._cevents(self.event_file_path,
                                                max_frames=10)
            cyf, cyt, cyy, cyx = event_list

            max_f = np.max(cyf)
            assert_equal(9, max_f)

    def test_event_same_as_detector(self):
        # the detector file should be the same as the event file
        # warnings filter for pixel size
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)

            orig_file = PlatypusNexus(
                os.path.join(self.path, 'PLP0011641.nx.hdf'))

        orig_det = orig_file.cat.detector
        frames = [np.arange(0, 501744)]
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
        event_list, fpos = _cevent._cevents(self.event_file_path,
                                            max_frames=10)
        f, t, y, x = event_list
        max_f = np.max(f)
        assert_equal(9, max_f)

    def test_values(self):
        # We know the values of all the events in the file from another program
        # test that a set of random events are correct.
        assert_equal(self.f[0], 0)
        assert_equal(self.t[0], 18912)
        assert_equal(self.y[0], 18)
        assert_equal(self.x[0], 19)

        assert_equal(self.f[-1], 501743)
        assert_equal(self.t[-1], 20786)
        assert_equal(self.y[-1], 16)
        assert_equal(self.x[-1], 13)

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
        assert_equal(detector[0, 300, 97, 0], 7)
        assert_equal(detector[1, 402, 93, 0], 1)
        assert_equal(detector[4, 509, 97, 0], 5)

        x_bins = np.array([210.5, -210.5])
        y_bins = np.linspace(110.5, -110.5, 222)
        t_bins = np.linspace(0, 50000, 1001)
        frames = [[6, 10, 12, 1000, 1001, 1002, 1003]]
        detector, fbins = event.process_event_stream(self.event_list,
                                                     frames,
                                                     t_bins,
                                                     y_bins,
                                                     x_bins)
        assert_equal(np.sum(detector.ravel()), 11)

        # # now see what happens if we go too far with the frame_bins
        # frames = event.framebins_to_frames([0, 24000, 30000])
        # detector, fbins = event.process_event_stream(self.event_list,
        #                                              frames,
        #                                              t_bins,
        #                                              y_bins,
        #                                              x_bins)
        # assert_equal(np.size(detector, 0), 1)

    def test_monobloc_events(self):
        # the event file changed when the ILL monobloc detector was installed
        event_file_path = os.path.join(self.path,
                                       'DAQ_2019-11-25T12-25-07',
                                       'DATASET_0',
                                       'EOS.bin')
        data = event.events(event_file_path)
        f, t, y, x = data[0]
        assert len(t) == 71223

        x_bins = np.array([2.5, 28.5])
        y_bins = np.linspace(-0.5, 1023.5, 1025)
        t_bins = np.linspace(0, 40000, 1001)
        frames = [np.arange(0, 57599, 1)]

        detector, fbins = event.process_event_stream(data[0],
                                                     frames,
                                                     t_bins,
                                                     y_bins,
                                                     x_bins)
        detector = np.squeeze(detector)

        with h5py.File(os.path.join(self.path, 'PLP0046853.nx.hdf'), 'r') as g:
            det = np.copy(g['entry1/data/hmm'])
            det = np.squeeze(det)
            assert det.shape == (1000, 1024)

        assert_equal(np.sum(detector, 1), np.sum(det, 1))
