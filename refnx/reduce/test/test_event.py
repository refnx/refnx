from os.path import join as pjoin
import os
import warnings
from collections import namedtuple

import numpy as np
import h5py
import pytest

from numpy.testing import assert_equal
import refnx.reduce.event as event
from refnx.reduce import PlatypusNexus

try:
    import refnx.reduce._cevent as _cevent
except ImportError:
    HAVE_CEVENTS = False
else:
    HAVE_CEVENTS = True


@pytest.fixture(scope="class")
def event_setup(data_directory):
    if data_directory is None:
        return None

    Event_Setup = namedtuple(
        "Event_Setup",
        [
            "data_directory",
            "event_file_path",
            "event_list",
            "fpos",
            "f",
            "t",
            "y",
            "x",
        ],
    )

    event_file_path = os.path.join(
        data_directory,
        "reduce",
        "DAQ_2012-01-20T21-50-32",
        "DATASET_0",
        "EOS.bin",
    )
    with open(event_file_path, "rb") as f:
        event_list, fpos = _cevent._cevents(f)

    f, t, y, x = event_list

    stp = Event_Setup(
        data_directory, event_file_path, event_list, fpos, f, t, y, x
    )
    return stp


class TestEvent:
    @pytest.mark.usefixtures("no_data_directory")
    def test_events_smoke(self, event_setup):
        # check that the event.events function works
        event.events(event_setup.event_file_path)

    @pytest.mark.usefixtures("no_data_directory")
    def test_num_events(self, event_setup):
        assert_equal(event_setup.x.size, 783982)

    @pytest.mark.usefixtures("no_data_directory")
    def test_max_frames(self, event_setup):
        # test reading only a certain number of frames

        if HAVE_CEVENTS:
            with open(event_setup.event_file_path, "rb") as g:
                event_list, fpos = _cevent._cevents(g, max_frames=10)
                cyf, cyt, cyy, cyx = event_list

            max_f = np.max(cyf)
            assert_equal(9, max_f)

            event_list, fpos = _cevent._cevents(
                event_setup.event_file_path, max_frames=10
            )
            cyf, cyt, cyy, cyx = event_list

            max_f = np.max(cyf)
            assert_equal(9, max_f)

    @pytest.mark.usefixtures("no_data_directory")
    def test_event_same_as_detector(self, event_setup):
        # the detector file should be the same as the event file
        # warnings filter for pixel size
        pth = pjoin(event_setup.data_directory, "reduce")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            orig_file = PlatypusNexus(pjoin(pth, "PLP0011641.nx.hdf"))

        orig_det = orig_file.cat.detector
        frames = [np.arange(0, 501744)]
        event_det, fb = event.process_event_stream(
            event_setup.event_list,
            frames,
            orig_file.cat.t_bins,
            orig_file.cat.y_bins,
            orig_file.cat.x_bins,
        )
        assert_equal(event_det, orig_det)

        # PlatypusNexus.process_event_stream should be the same as well
        det, fc, bm = orig_file.process_event_stream(frame_bins=[])
        assert_equal(det, orig_det)

    @pytest.mark.usefixtures("no_data_directory")
    def test_open_with_path(self, event_setup):
        # give the event reader a file path
        event_list, fpos = _cevent._cevents(
            event_setup.event_file_path, max_frames=10
        )
        f, t, y, x = event_list
        max_f = np.max(f)
        assert_equal(9, max_f)

    @pytest.mark.usefixtures("no_data_directory")
    def test_values(self, event_setup):
        # We know the values of all the events in the file from another program
        # test that a set of random events are correct.
        assert_equal(event_setup.f[0], 0)
        assert_equal(event_setup.t[0], 18912)
        assert_equal(event_setup.y[0], 18)
        assert_equal(event_setup.x[0], 19)

        assert_equal(event_setup.f[-1], 501743)
        assert_equal(event_setup.t[-1], 20786)
        assert_equal(event_setup.y[-1], 16)
        assert_equal(event_setup.x[-1], 13)

    @pytest.mark.usefixtures("no_data_directory")
    def test_process_event_stream(self, event_setup):
        x_bins = np.array([60.5, -60.5])
        y_bins = np.linspace(110.5, -110.5, 222)
        t_bins = np.linspace(0, 50000, 1001)

        frames = event.framebins_to_frames(np.linspace(0, 24000, 7))
        detector, fbins = event.process_event_stream(
            event_setup.event_list, frames, t_bins, y_bins, x_bins
        )
        assert_equal(detector[0, 300, 97, 0], 7)
        assert_equal(detector[1, 402, 93, 0], 1)
        assert_equal(detector[4, 509, 97, 0], 5)

        x_bins = np.array([210.5, -210.5])
        y_bins = np.linspace(110.5, -110.5, 222)
        t_bins = np.linspace(0, 50000, 1001)
        frames = [[6, 10, 12, 1000, 1001, 1002, 1003]]
        detector, fbins = event.process_event_stream(
            event_setup.event_list, frames, t_bins, y_bins, x_bins
        )
        assert_equal(np.sum(detector.ravel()), 11)

        # # now see what happens if we go too far with the frame_bins
        # frames = event.framebins_to_frames([0, 24000, 30000])
        # detector, fbins = event.process_event_stream(self.event_list,
        #                                              frames,
        #                                              t_bins,
        #                                              y_bins,
        #                                              x_bins)
        # assert_equal(np.size(detector, 0), 1)

    @pytest.mark.usefixtures("no_data_directory")
    def test_monobloc_events(self, event_setup):
        # the event file changed when the ILL monobloc detector was installed
        pth = pjoin(event_setup.data_directory, "reduce")

        event_file_path = pjoin(
            pth, "DAQ_2019-11-25T12-25-07", "DATASET_0", "EOS.bin"
        )
        data = event.events(event_file_path)
        f, t, y, x = data[0]
        assert len(t) == 71223

        x_bins = np.array([2.5, 28.5])
        y_bins = np.linspace(-0.5, 1023.5, 1025)
        t_bins = np.linspace(0, 40000, 1001)
        frames = [np.arange(0, 57599, 1)]

        detector, fbins = event.process_event_stream(
            data[0], frames, t_bins, y_bins, x_bins
        )
        detector = np.squeeze(detector)

        with h5py.File(pjoin(pth, "PLP0046853.nx.hdf"), "r") as g:
            det = np.copy(g["entry1/data/hmm"])
            det = np.squeeze(det)
            assert det.shape == (1000, 1024)

        assert_equal(np.sum(detector, 1), np.sum(det, 1))
