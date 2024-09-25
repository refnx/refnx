from pathlib import Path
import mmap
import os
import warnings
from collections import namedtuple

import numpy as np
import h5py
import pytest

from numpy.testing import assert_equal
import refnx.reduce.event as event
from refnx.reduce import PlatypusNexus, SpatzNexus

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
            "nexus_file_path",
            "ReflectNexus",
            "event_list",
            "fpos",
            "f",
            "t",
            "y",
            "x",
        ],
    )

    # PLP example
    event_file_path = Path(
        data_directory,
        "reduce",
        "DAQ_2022-12-02T07-26-10",
        "DATASET_0",
        "EOS.bin",
    )
    nexus_file_path = Path(data_directory, "reduce", "PLP0057863.nx.hdf")
    with open(event_file_path, "rb") as f:
        event_list, fpos = _cevent._cevents(f)

    f, t, y, x = event_list

    stp_PLP = Event_Setup(
        data_directory,
        event_file_path,
        nexus_file_path,
        PlatypusNexus,
        event_list,
        fpos,
        f,
        t,
        y,
        x,
    )

    # SPZ example
    event_file_path = Path(
        data_directory,
        "reduce",
        "DAQ_2022-12-02T14-30-47",
        "DATASET_0",
        "EOS.bin",
    )
    nexus_file_path = Path(data_directory, "reduce", "SPZ0008054.nx.hdf")
    with open(event_file_path, "rb") as f:
        event_list, fpos = _cevent._cevents(f)

    f, t, y, x = event_list

    stp_SPZ = Event_Setup(
        data_directory,
        event_file_path,
        nexus_file_path,
        SpatzNexus,
        event_list,
        fpos,
        f,
        t,
        y,
        x,
    )

    return stp_PLP, stp_SPZ


class TestEvent:
    def test_events_smoke(self, event_setup):
        # check that the event.events function works
        # PLP
        for p in event_setup:
            event.events(p.event_file_path)

    def test_num_events(self, event_setup):
        assert_equal(event_setup[0].x.size, 2209769)
        assert_equal(event_setup[1].x.size, 431693)

    def test_max_frames(self, event_setup):
        # test reading only a certain number of frames

        if HAVE_CEVENTS:
            with open(event_setup[0].event_file_path, "rb") as g:
                event_list, fpos = _cevent._cevents(g, max_frames=1111)
                cyf, cyt, cyy, cyx = event_list

            assert np.max(cyf) < 1111

            event_list, fpos = _cevent._cevents(
                event_setup[1].event_file_path, max_frames=1111
            )
            cyf, cyt, cyy, cyx = event_list
            assert np.max(cyf) < 1111

    def test_event_same_as_detector(self, event_setup):
        # the detector file should be the same as the event file
        # warnings filter for pixel size

        for evt in event_setup:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                orig_file = evt.ReflectNexus(evt.nexus_file_path)

            orig_det = orig_file.cat.detector
            frames = [np.arange(0, np.max(evt.f) + 1)]
            event_det, fb = event.process_event_stream(
                evt.event_list,
                frames,
                orig_file.cat.t_bins,
                orig_file.cat.y_bins,
                orig_file.cat.x_bins,
            )
            assert_equal(event_det, orig_det)
            #
            # # PlatypusNexus.process_event_stream should be the same as well
            # det, fc, bm = orig_file.process_event_stream(frame_bins=[])
            # assert_equal(det, orig_det)

    def test_values(self, event_setup):
        # We know the values of all the events in the file from another program
        # test that a set of random events are correct.
        assert_equal(event_setup[0].f[0], 0)
        assert_equal(event_setup[0].t[0], 6904)
        assert_equal(event_setup[0].y[0], 466)
        assert_equal(event_setup[0].x[0], 22)

        assert_equal(event_setup[0].f[-1], 118798)
        assert_equal(event_setup[0].t[-1], 16482)
        assert_equal(event_setup[0].y[-1], 472)
        assert_equal(event_setup[0].x[-1], 16)

    def test_monobloc_events(self, event_setup):
        # the event file changed when the ILL monobloc detector was installed
        data = event.events(event_setup[0].event_file_path)
        f, t, y, x = data[0]
        assert len(t) == 2209769

        # in the nexus setup dcr used
        x_bins = np.array([9.5, 19.5])
        y_bins = np.linspace(-0.5, 1023.5, 1025)
        t_bins = np.linspace(0, 30000, 1001)
        frames = [np.arange(0, 118800, 1)]

        detector, fbins = event.process_event_stream(
            data[0], frames, t_bins, y_bins, x_bins
        )

        with h5py.File(event_setup[0].nexus_file_path, "r") as g:
            orig_det = np.copy(g["entry1/data/hmm"])
            assert orig_det.shape == (1, 1000, 1024, 1)

        assert_equal(detector, orig_det)

    def test_clock_scale(self, event_setup):
        # the event file changed when the ILL monobloc detector was installed
        pth = Path(event_setup[0].event_file_path)

        # PLP clock_scale is 16 ns per tick
        with open(pth, "rb") as f:
            buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        hdr_base, hdr_packing = _cevent.event_header(buffer)
        assert hdr_base.clock_scale == 16

        # SPZ clock_scale is 100 ns per tick
        pth = Path(event_setup[1].event_file_path)
        with open(pth, "rb") as f:
            buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        hdr_base, hdr_packing = _cevent.event_header(buffer)
        assert hdr_base.clock_scale == 100
