import os
import numbers
import warnings
import pickle
from pathlib import Path

import pytest
import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_array_less,
    assert_allclose,
)
import h5py

import refnx.reduce.platypusnexus as plp
from refnx.reduce import (
    PlatypusReduce,
    PlatypusNexus,
    SpatzNexus,
    basename_datafile,
    catalogue,
    SpinSet,
)
from refnx.reduce.peak_utils import gauss
from refnx.reduce.platypusnexus import (
    fore_back_region,
    EXTENT_MULT,
    PIXEL_OFFSET,
    create_detector_norm,
    Catalogue,
    PlatypusCatalogue,
    create_reflect_nexus,
    ReductionOptions,
)
from refnx.reflect import SpinChannel


class TestSpinSet(object):
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path, data_directory):
        self.pth = data_directory / "reduce" / "PNR_files"

        def fpath(f):
            return self.pth / f

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            self.spinset = SpinSet(
                down_down=fpath("PLP0008864.nx.hdf"),
                up_up=fpath("PLP0008861.nx.hdf"),
                down_up=fpath("PLP0008863.nx.hdf"),
                up_down=fpath("PLP0008862.nx.hdf"),
            )

            self.spinset_3 = SpinSet(
                down_down=fpath("PLP0008864.nx.hdf"),
                up_up=fpath("PLP0008861.nx.hdf"),
                down_up=fpath("PLP0008863.nx.hdf"),
            )
            self.spinset_2 = SpinSet(
                down_down=fpath("PLP0008864.nx.hdf"),
                up_up=fpath("PLP0008861.nx.hdf"),
            )

        self.cwd = Path.cwd()

        self.tmp_path = tmp_path
        os.chdir(self.tmp_path)
        return 0

    def test_spin_channels(self):
        assert self.spinset.spin_channels == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert self.spinset_3.spin_channels == [(0, 0), (0, 1), None, (1, 1)]
        assert self.spinset_2.spin_channels == [(0, 0), None, None, (1, 1)]

    def test_process(self):
        self.spinset.process()
        assert self.spinset.dd.processed_spectrum
        assert self.spinset.du.processed_spectrum
        assert self.spinset.ud.processed_spectrum
        assert self.spinset.uu.processed_spectrum

        self.spinset_3.process()
        assert self.spinset_3.dd.processed_spectrum
        assert self.spinset_3.du.processed_spectrum
        assert self.spinset_3.ud is None
        assert self.spinset_3.uu.processed_spectrum

        self.spinset_2.process()
        assert self.spinset_2.dd.processed_spectrum
        assert self.spinset_2.du is None
        assert self.spinset_2.ud is None
        assert self.spinset_2.uu.processed_spectrum

    def test_processing_different_reduction_options(self):
        # Check every combination of spin channel and reduction option that
        # determines the resulting wavelength axis to make sure errors
        # are raised appropriately

        standard_opts = dict(
            lo_wavelength=2.5,
            hi_wavelength=12.5,
            rebin_percent=3,
        )
        for spin_set in [self.spinset, self.spinset_3, self.spinset_2]:
            for sc in ["dd", "du", "ud", "uu"]:
                if spin_set.channels[sc] is None:
                    continue
                else:
                    for option in [
                        "lo_wavelength",
                        "hi_wavelength",
                        "rebin_percent",
                    ]:
                        spin_set.sc_opts[sc].update({option: 5})
                        with pytest.raises(ValueError):
                            spin_set.process()
                        # Reset dd_opts
                        spin_set.sc_opts[sc].update(standard_opts)


class TestPlatypusNexus(object):
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path, data_directory):
        self.pth = data_directory / "reduce"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.f113 = PlatypusNexus(self.pth / "PLP0011613.nx.hdf")

            # to ensure that file can be opened with a Path
            pth = self.pth / "PLP0011641.nx.hdf"
            self.f641 = PlatypusNexus(pth)

            # These PNR datasets all have different flipper settings
            self.f8861 = PlatypusNexus(
                self.pth / "PNR_files/PLP0008861.nx.hdf"
            )
            self.f8862 = PlatypusNexus(
                self.pth / "PNR_files/PLP0008862.nx.hdf"
            )
            self.f8863 = PlatypusNexus(
                self.pth / "PNR_files/PLP0008863.nx.hdf"
            )
            self.f8864 = PlatypusNexus(
                self.pth / "PNR_files/PLP0008864.nx.hdf"
            )

        self.cwd = Path.cwd()

        self.tmp_path = tmp_path
        os.chdir(self.tmp_path)
        return 0

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_chod(self):
        flight_length = self.f113.chod()
        assert_almost_equal(flight_length[0], 7141.413818359375)
        assert_almost_equal(flight_length[1], 808)

        flight_length = self.f641.chod(omega=1.8, twotheta=3.6)
        assert_almost_equal(flight_length[0], 7146.3567785516016)
        assert_almost_equal(flight_length[1], 808)

    def test_phase_angle(self):
        # TODO. Add more tests where the phase_angle isn't zero.
        phase_angle, master_opening = self.f641.phase_angle()
        assert_almost_equal(phase_angle, 0)
        assert_almost_equal(master_opening, 60)

        assert self.f641.cat.t_offset is None

    def test_background_subtract_line(self):
        # checked each step of the background subtraction with IGOR
        # so this test background correction should be correct.

        # create some test data
        xvals = np.linspace(-10, 10, 201)
        yvals = np.ceil(gauss(xvals, 0, 100, 0, 1) + 2 * xvals + 30)

        # add some reproducible random noise
        np.random.seed(1)
        yvals += np.sqrt(yvals) * np.random.randn(yvals.size)
        yvals_sd = np.sqrt(yvals)

        mask = np.zeros(201, bool)
        mask[30:70] = True
        mask[130:160] = True

        profile, profile_sd = plp.background_subtract_line(
            yvals, yvals_sd, mask
        )

        verified_data = np.load(self.pth / "background_subtract.npy")

        assert_almost_equal(verified_data, np.c_[profile, profile_sd])

    def test_find_specular_ridge(self):
        xvals = np.linspace(-10, 10, 201)
        yvals = np.ceil(gauss(xvals, 0, 1000, 0, 1))
        detector = np.repeat(yvals[:, np.newaxis], 1000, axis=1).T
        detector_sd = np.sqrt(detector)
        output = plp.find_specular_ridge(
            detector[np.newaxis, :], detector_sd[np.newaxis, :]
        )
        assert len(output) == 5
        assert_almost_equal(output[0][0], 100)

    def test_pickle(self):
        # can we pickle a PlatypusNexus object?
        self.f8864.process()
        pkl = pickle.dumps(self.f8864)
        o = pickle.loads(pkl)
        ops, f8864 = o.processed_spectrum, self.f8864.processed_spectrum
        assert_allclose(ops["m_spec"], f8864["m_spec"])

    def test_background_subtract(self):
        # create some test data
        xvals = np.linspace(-10, 10, 201)
        yvals = np.ceil(gauss(xvals, 0, 100, 0, 1) + 2 * xvals + 30)

        # add some reproducible random noise
        np.random.seed(1)
        yvals += np.sqrt(yvals) * np.random.randn(yvals.size)
        yvals_sd = np.sqrt(yvals)

        # now make an (N, T, Y) detector image
        n_tbins = 10
        detector = np.repeat(yvals, n_tbins).reshape(xvals.size, n_tbins).T
        detector_sd = (
            np.repeat(yvals_sd, n_tbins).reshape(xvals.size, n_tbins).T
        )
        detector = detector.reshape(1, n_tbins, xvals.size)
        detector_sd = detector_sd.reshape(1, n_tbins, xvals.size)

        mask = np.zeros((1, n_tbins, 201), bool)
        mask[:, :, 30:70] = True
        mask[:, :, 130:160] = True

        det_bkg, detSD_bkg = plp.background_subtract(
            detector, detector_sd, mask
        )

        # each of the (N, T) entries should have the same background subtracted
        # entries
        verified_data = np.load(self.pth / "background_subtract.npy")

        it = np.nditer(detector, flags=["multi_index"])
        it.remove_axis(2)
        while not it.finished:
            profile = det_bkg[it.multi_index]
            profile_sd = detSD_bkg[it.multi_index]
            assert_almost_equal(verified_data, np.c_[profile, profile_sd])
            it.iternext()

    def test_calculate_bins(self):
        bins = plp.calculate_wavelength_bins(2.0, 18, 2.0)
        assert_almost_equal(bins[0], 1.98)
        assert_almost_equal(bins[-1], 18.18)

    def test_event(self):
        # When you use event mode processing, make sure the right amount of
        # spectra are created
        out = self.f641.process(eventmode=[0, 900, 1800], integrate=0)
        assert np.size(out[1], axis=0) == 2

    def test_event_folder(self):
        self.f641.process(
            eventmode=[0, 900, 1800], integrate=0, event_folder=self.pth
        )
        pth = Path(self.pth)
        self.f641.process(
            eventmode=[0, 900, 1800], integrate=0, event_folder=pth
        )

    def test_multiple_acquisitions(self):
        """
        TODO: add a dataset which has multiple spectra in it, and make sure it
        processes.
        """
        pass

    def test_reduction_runs(self):
        # just check it runs
        self.f641.process()

        # check that event mode reduction gives the same output as non-event
        # mode reduction.
        spectrum0 = self.f641.process(direct=False)
        assert "reduction_options" in self.f641.processed_spectrum

        spectrum1 = self.f641.process(direct=False, eventmode=[], integrate=0)
        assert_allclose(spectrum0[1][0], spectrum1[1][0], rtol=0.001)

        # check that the wavelength resolution is roughly right, between 7 and
        # 8%.
        res = (
            self.f641.processed_spectrum["m_lambda_fwhm"][0]
            / self.f641.processed_spectrum["m_lambda"][0]
        )
        assert_array_less(res, np.ones_like(res) * 0.08)
        assert_array_less(np.ones_like(res) * 0.07, res)

    def test_save_spectrum(self):
        # test saving spectrum
        self.f113.process()

        # can save the spectra by supplying a filename
        self.f113.write_spectrum_xml(self.tmp_path / "test.xml")
        self.f113.write_spectrum_dat(self.tmp_path / "test.dat")

        # can save by supplying file handle:
        with open(self.tmp_path / "test.xml", "wb") as f:
            self.f113.write_spectrum_xml(f)

    def test_accumulate_files(self):
        fnames = ["PLP0000708.nx.hdf", "PLP0000709.nx.hdf"]
        pths = [self.pth / fname for fname in fnames]
        plp.accumulate_HDF_files(pths)
        f8, f9, fadd = None, None, None

        try:
            f8 = h5py.File(self.pth / "PLP0000708.nx.hdf", "r")
            f9 = h5py.File(self.pth / "PLP0000709.nx.hdf", "r")
            fadd = h5py.File(self.tmp_path / "ADD_PLP0000708.nx.hdf", "r")

            f8d = f8["entry1/data/hmm"][0]
            f9d = f9["entry1/data/hmm"][0]
            faddd = fadd["entry1/data/hmm"][0]
            assert_equal(faddd, f8d + f9d)
        finally:
            if f8 is not None:
                f8.close()
            if f9 is not None:
                f9.close()
            if fadd is not None:
                fadd.close()

    def test_accumulate_files_reduce(self):
        # test by adding a file to itself. Should have smaller stats
        fnames = ["PLP0000708.nx.hdf", "PLP0000708.nx.hdf"]
        pths = [self.pth / fname for fname in fnames]
        plp.accumulate_HDF_files(pths)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # it should be processable
            fadd = PlatypusNexus(Path.cwd() / "ADD_PLP0000708.nx.hdf")
            fadd.process()

            # it should also be reduceable
            reducer = PlatypusReduce(self.pth / "PLP0000711.nx.hdf")

            datasets, reduced = reducer.reduce(
                Path.cwd() / "ADD_PLP0000708.nx.hdf"
            )
            assert "y" in reduced

            # the error bars should be smaller
            datasets2, reduced2 = reducer.reduce(
                self.pth / "PLP0000708.nx.hdf"
            )

            assert np.all(reduced["y_err"] < reduced2["y_err"])

    def test_manual_beam_find(self):
        # you can specify a function that finds where the specular ridge is.
        def manual_beam_find(detector, detector_sd, name):
            beam_centre = np.zeros(len(detector))
            beam_sd = np.zeros(len(detector))
            beam_centre += 50
            beam_sd += 5
            return beam_centre, beam_sd, np.array([40]), np.array([60]), [[]]

        # the manual beam find is only mandatory when peak_pos == -1.
        # # the beam_sd is much larger than the beam divergence, so a warning
        # # should be raised.
        # with pytest.warns(UserWarning):
        self.f113.process(manual_beam_find=manual_beam_find, peak_pos=-1)
        assert_equal(self.f113.processed_spectrum["m_beampos"][0], 50)

        # manual beam finding also specifies the lower and upper pixel of the
        # foreground
        assert_equal(self.f113.processed_spectrum["lopx"][0], 40)
        assert_equal(self.f113.processed_spectrum["hipx"][0], 60)

    def test_fore_back_region(self):
        # calculation of foreground and background regions is done correctly
        centres = np.array([100.0, 90.0])
        sd = np.array([5.0, 11.5])
        lopx, hipx, background_pixels = fore_back_region(centres, sd)

        assert len(lopx) == 2
        assert len(hipx) == 2
        assert len(background_pixels) == 2
        assert isinstance(lopx[0], numbers.Integral)
        assert isinstance(hipx[0], numbers.Integral)

        calc_lower = np.floor(centres - sd * EXTENT_MULT)
        assert_equal(lopx, calc_lower)
        calc_higher = np.ceil(centres + sd * EXTENT_MULT)
        assert_equal(hipx, calc_higher)

        y1 = np.atleast_1d(np.round(lopx - PIXEL_OFFSET).astype("int"))
        y0 = np.atleast_1d(
            np.round(lopx - PIXEL_OFFSET - EXTENT_MULT * sd).astype("int")
        )

        y2 = np.atleast_1d(np.round(hipx + PIXEL_OFFSET).astype("int"))
        y3 = np.atleast_1d(
            np.round(hipx + PIXEL_OFFSET + EXTENT_MULT * sd).astype("int")
        )

        bp = np.r_[np.arange(y0[0], y1[0] + 1), np.arange(y2[0], y3[0] + 1)]

        assert_equal(bp, background_pixels[0])

    def test_lopx_hipx(self):
        # we should be able to specify the exact pixel numbers we want to
        # integrate
        rdo = ReductionOptions(
            lopx_hipx=(50, 60), background=False, peak_pos=(55, 2)
        )
        _, m_spec, _ = self.f113.process(**rdo)

        m_topandtail = self.f113.m_topandtail
        assert self.f113.lopx[0] == 50
        assert self.f113.hipx[0] == 60
        check = np.sum(m_topandtail[..., 50:61], axis=-1)[0]
        assert_allclose(m_spec[0], check)

    def test_basename_datafile(self):
        # check that the right basename is returned
        pth = "a/b/c.nx.hdf"
        assert basename_datafile(pth) == "c"

        pth = "c.nx.hdf"
        assert basename_datafile(pth) == "c"

    def test_floodfield_correction(self):
        # check that flood field calculation works
        # the values were worked out by hand on a randomly
        # generated array
        test_norm = np.array(
            [1.12290503, 1.23743017, 0.8603352, 0.70111732, 1.07821229]
        )
        test_norm_sd = np.array(
            [0.05600541, 0.05879208, 0.04902215, 0.04425413, 0.05487956]
        )

        fname = self.pth / "flood.h5"
        with h5py.File(fname, "r") as f:
            norm, norm_sd = create_detector_norm(f, 3.5, -3.5, axis=3)

            assert_almost_equal(norm, test_norm, 6)
            assert_almost_equal(norm_sd, test_norm_sd, 6)

            norm, norm_sd = create_detector_norm(f, -3.5, 3.5, axis=3)
            assert_almost_equal(norm, test_norm, 6)
            assert_almost_equal(norm_sd, test_norm_sd, 6)

    def test_PNR_metadata(self):
        self.f8861.process()
        self.f8862.process()
        self.f8863.process()
        self.f8864.process()

        # Check that we can read all of the flipper statuses in files

        # Flipper 1 on, flipper 2 on
        assert_almost_equal(self.f8861.cat.cat["pol_flip_current"], 5.0)
        assert_almost_equal(self.f8861.cat.cat["anal_flip_current"], 4.5)

        # Flipper 1 on, flipper 2 off
        assert_almost_equal(self.f8862.cat.cat["anal_flip_current"], 0)
        assert_almost_equal(self.f8862.cat.cat["pol_flip_current"], 5.0)

        # Flipper 1 off, flipper 2 on
        assert_almost_equal(self.f8863.cat.cat["anal_flip_current"], 4.5)
        assert_almost_equal(self.f8863.cat.cat["pol_flip_current"], 0)

        # Flipper 1 off, flipper 2 off
        assert_almost_equal(self.f8864.cat.cat["anal_flip_current"], 0)
        assert_almost_equal(self.f8864.cat.cat["pol_flip_current"], 0)

        # Check SpinChannel for each file
        assert self.f8861.spin_state == SpinChannel.UP_UP
        assert self.f8862.spin_state == SpinChannel.UP_DOWN
        assert self.f8863.spin_state == SpinChannel.DOWN_UP
        assert self.f8864.spin_state == SpinChannel.DOWN_DOWN

        # test spin channel setting
        # non spin analysed. mode is POL, not POLANAL
        pn = PlatypusNexus(self.pth / "PLP0016427.nx.hdf")
        assert pn.spin_state == SpinChannel.UP_UP
        pn = PlatypusNexus(self.pth / "PLP0016426.nx.hdf")
        assert pn.spin_state == SpinChannel.DOWN_DOWN

    def test_PNR_magnet_read(self):
        self.f8861.process()
        # Check magnetic field sensors
        assert_almost_equal(self.f8861.cat.cat["magnet_current_set"], 0)
        assert_almost_equal(self.f8861.cat.cat["magnet_output_current"], 0.001)


class TestSpatzNexus:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path, data_directory):
        self.pth = data_directory / "reduce"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pth = self.pth / "SPZ0000342.nx.hdf"
            self.f342 = SpatzNexus(pth)
        self.cwd = Path.cwd()

        self.tmp_path = tmp_path
        os.chdir(self.tmp_path)
        return 0

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_chod(self):
        flight_length = self.f342.chod()
        flight_length = self.f342.chod()
        cat = self.f342.cat
        assert_almost_equal(cat.sample_distance, 6237.0)
        assert_almost_equal(flight_length[0], 8062.0232, decimal=4)
        assert_almost_equal(flight_length[1], 479.9536, decimal=4)

    def test_phase_angle(self):
        assert_allclose(self.f342.cat.master_phase_offset, -25.90)
        assert self.f342.cat.master == 1
        assert self.f342.cat.slave == 2
        assert self.f342.cat.t_offset is None
        assert_allclose(self.f342.cat.frequency, 25)
        assert_allclose(self.f342.cat.phase, 34.22)
        assert_allclose(self.f342.cat.poff_c2_slave_1_master[0], -0.22)

        phase_angle, master_opening = self.f342.phase_angle()
        # chopper 1 has an opening of 26 degrees
        assert_almost_equal(master_opening, 26)
        assert_allclose(phase_angle, 0, atol=1e-5)

        toff = self.f342.time_offset(-25.90, 26, 25, 0.0, None, None, None)
        assert_allclose(toff, 5.5555555555555)

        toff = self.f342.time_offset(
            -25.90,
            26,
            25,
            0.0,
            None,
            None,
            None,
            t_offset=1438.8888888888888,
        )
        assert_allclose(toff, 0, atol=1e-12)

    def test_detector_translation(self):
        f = SpatzNexus(self.pth / "SPZ0012268.nx.hdf")
        dy = f.cat.cat["dy"]
        assert_allclose(dy, 864.0137)


def test_catalogue(data_directory):
    pth = Path(data_directory) / "reduce"
    catalogue(0, 10000000, data_folder=pth, prefix="PLP")
    catalogue(0, 10000000, data_folder=pth, prefix="SPZ")

    pth = Path(data_directory) / "reduce"
    catalogue(0, 10000000, data_folder=pth, prefix="PLP")

    # check that you can supply a sequence of keys to return
    df = catalogue(0, 1000000, data_folder=pth, keys=["omega"])
    assert df.columns.size == 1
    assert df.columns[0] == "omega"


def test_Catalogue_pickle(data_directory):
    pth = Path(data_directory) / "reduce"
    f113 = pth / "PLP0011613.nx.hdf"
    with h5py.File(f113) as f:
        c = PlatypusCatalogue(f)

    assert c.datafile_number == 11613
    assert c.prefix == "PLP"
    pkl = pickle.dumps(c)
    pickle.loads(pkl)


def test_create_nexus(data_directory):
    pth = Path(data_directory) / "reduce"
    f113 = pth / "PLP0011613.nx.hdf"
    f = create_reflect_nexus(f113)
    assert isinstance(f, PlatypusNexus)
