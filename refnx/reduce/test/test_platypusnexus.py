import os
import numbers
import warnings

import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
                           assert_array_less, assert_allclose)
import h5py

import refnx.reduce.platypusnexus as plp
from refnx.reduce import PlatypusReduce, PlatypusNexus, basename_datafile
from refnx.reduce.peak_utils import gauss
from refnx.reduce.platypusnexus import (fore_back_region, EXTENT_MULT,
                                        PIXEL_OFFSET, create_detector_norm)


class TestPlatypusNexus(object):

    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir):
        self.pth = os.path.dirname(os.path.abspath(__file__))

        self.f113 = PlatypusNexus(os.path.join(self.pth,
                                               'PLP0011613.nx.hdf'))
        self.f641 = PlatypusNexus(os.path.join(self.pth,
                                               'PLP0011641.nx.hdf'))
        self.cwd = os.getcwd()

        self.tmpdir = tmpdir.strpath
        os.chdir(self.tmpdir)
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
        assert_almost_equal(master_opening, 1.04719755)

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

        mask = np.zeros(201, np.bool)
        mask[30:70] = True
        mask[130:160] = True

        profile, profile_sd = plp.background_subtract_line(yvals,
                                                           yvals_sd,
                                                           mask)

        verified_data = np.load(os.path.join(self.pth,
                                             'background_subtract.npy'))

        assert_almost_equal(verified_data, np.c_[profile, profile_sd])

    def test_find_specular_ridge(self):
        xvals = np.linspace(-10, 10, 201)
        yvals = np.ceil(gauss(xvals, 0, 1000, 0, 1))
        detector = np.repeat(yvals[:, np.newaxis], 1000, axis=1).T
        detector_sd = np.sqrt(detector)
        output = plp.find_specular_ridge(detector[np.newaxis, :],
                                         detector_sd[np.newaxis, :])
        assert_(len(output) == 5)
        assert_almost_equal(output[0][0], 100)

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
        detector = np.repeat(yvals,
                             n_tbins).reshape(xvals.size, n_tbins).T
        detector_sd = np.repeat(yvals_sd,
                                n_tbins).reshape(xvals.size, n_tbins).T
        detector = detector.reshape(1, n_tbins, xvals.size)
        detector_sd = detector_sd.reshape(1, n_tbins, xvals.size)

        mask = np.zeros((1, n_tbins, 201), np.bool)
        mask[:, :, 30:70] = True
        mask[:, :, 130:160] = True

        det_bkg, detSD_bkg = plp.background_subtract(detector,
                                                     detector_sd,
                                                     mask)

        # each of the (N, T) entries should have the same background subtracted
        # entries
        verified_data = np.load(os.path.join(self.pth,
                                             'background_subtract.npy'))

        it = np.nditer(detector, flags=['multi_index'])
        it.remove_axis(2)
        while not it.finished:
            profile = det_bkg[it.multi_index]
            profile_sd = detSD_bkg[it.multi_index]
            assert_almost_equal(verified_data, np.c_[profile, profile_sd])
            it.iternext()

    def test_calculate_bins(self):
        bins = plp.calculate_wavelength_bins(2., 18, 2.)
        assert_almost_equal(bins[0], 1.98)
        assert_almost_equal(bins[-1], 18.18)

    def test_event(self):
        # When you use event mode processing, make sure the right amount of
        # spectra are created
        out = self.f113.process(eventmode=[0, 900, 1800], integrate=0)
        assert_(np.size(out[1], axis=0) == 2)

    def test_event_folder(self):
        # When you use event mode processing, make sure the right amount of
        # spectra are created
        self.f113.process(eventmode=[0, 900, 1800], integrate=0,
                          event_folder=self.pth)

    def test_multiple_acquisitions(self):
        """
        TODO: add a dataset which has multiple spectra in it, and make sure it
        processes.
        """
        pass

    def test_reduction_runs(self):
        # just check it runs
        self.f113.process()

        # check that event mode reduction gives the same output as non-event
        # mode reduction.
        spectrum0 = self.f113.process(direct=True)
        spectrum1 = self.f113.process(direct=True, eventmode=[], integrate=0)
        assert_allclose(spectrum0[1][0], spectrum1[1][0], rtol=0.001)

        # check that the wavelength resolution is roughly right, between 7 and
        # 8%.
        res = (self.f113.processed_spectrum['m_lambda_fwhm'][0] /
               self.f113.processed_spectrum['m_lambda'][0])
        assert_array_less(res, np.ones_like(res) * 0.08)
        assert_array_less(np.ones_like(res) * 0.07, res)

    def test_save_spectrum(self):
        # test saving spectrum
        self.f113.process()

        # can save the spectra by supplying a filename
        self.f113.write_spectrum_xml(os.path.join(self.tmpdir, 'test.xml'))
        self.f113.write_spectrum_dat(os.path.join(self.tmpdir, 'test.dat'))

        # can save by supplying file handle:
        with open(os.path.join(self.tmpdir, 'test.xml'), 'wb') as f:
            self.f113.write_spectrum_xml(f)

    def test_accumulate_files(self):
        fnames = ['PLP0000708.nx.hdf', 'PLP0000709.nx.hdf']
        pths = [os.path.join(self.pth, fname) for fname in fnames]
        plp.accumulate_HDF_files(pths)
        f8, f9, fadd = None, None, None

        try:
            f8 = h5py.File(os.path.join(self.pth,
                                        'PLP0000708.nx.hdf'), 'r')
            f9 = h5py.File(os.path.join(self.pth,
                                        'PLP0000709.nx.hdf'), 'r')
            fadd = h5py.File(os.path.join(self.tmpdir,
                                          'ADD_PLP0000708.nx.hdf'), 'r')

            f8d = f8['entry1/data/hmm'][0]
            f9d = f9['entry1/data/hmm'][0]
            faddd = fadd['entry1/data/hmm'][0]
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
        fnames = ['PLP0000708.nx.hdf', 'PLP0000708.nx.hdf']
        pths = [os.path.join(self.pth, fname) for fname in fnames]
        plp.accumulate_HDF_files(pths)

        # it should be processable
        fadd = PlatypusNexus(os.path.join(os.getcwd(),
                                          'ADD_PLP0000708.nx.hdf'))
        fadd.process()

        # it should also be reduceable
        reducer = PlatypusReduce(os.path.join(self.pth,
                                              'PLP0000711.nx.hdf'))
        datasets, reduced = reducer.reduce(os.path.join(os.getcwd(),
                                           'ADD_PLP0000708.nx.hdf'))
        assert_('y' in reduced)

        # the error bars should be smaller
        datasets2, reduced2 = reducer.reduce(os.path.join(self.pth,
                                                          'PLP0000708.nx.hdf'))

        assert_(np.all(reduced['y_err'] < reduced2['y_err']))

    def test_manual_beam_find(self):
        # you can specify a function that finds where the specular ridge is.
        def manual_beam_find(detector, detector_sd):
            beam_centre = np.zeros(len(detector))
            beam_sd = np.zeros(len(detector))
            beam_centre += 50
            beam_sd += 5
            return beam_centre, beam_sd, np.array([40]), np.array([60]), [[]]

        # the manual beam find is only mandatory when peak_pos == -1.
        # the beam_sd is much larger than the beam divergence, so a warning
        # should be raised.
        with pytest.warns(UserWarning):
            self.f113.process(manual_beam_find=manual_beam_find,
                              peak_pos=-1)
        assert_equal(self.f113.processed_spectrum['m_beampos'][0], 50)

        # manual beam finding also specifies the lower and upper pixel of the
        # foreground
        assert_equal(self.f113.processed_spectrum['lopx'][0], 40)
        assert_equal(self.f113.processed_spectrum['hipx'][0], 60)

    def test_fore_back_region(self):
        # calculation of foreground and background regions is done correctly
        centres = np.array([100., 90.])
        sd = np.array([5., 11.5])
        lopx, hipx, background_pixels = fore_back_region(centres, sd)

        assert_(len(lopx) == 2)
        assert_(len(hipx) == 2)
        assert_(len(background_pixels) == 2)
        assert_(isinstance(lopx[0], numbers.Integral))
        assert_(isinstance(hipx[0], numbers.Integral))

        calc_lower = np.floor(centres - sd * EXTENT_MULT)
        assert_equal(lopx, calc_lower)
        calc_higher = np.ceil(centres + sd * EXTENT_MULT)
        assert_equal(hipx, calc_higher)

        y1 = np.atleast_1d(
            np.round(lopx - PIXEL_OFFSET).astype('int'))
        y0 = np.atleast_1d(
            np.round(lopx - PIXEL_OFFSET - EXTENT_MULT * sd).astype('int'))

        y2 = np.atleast_1d(
            np.round(hipx + PIXEL_OFFSET).astype('int'))
        y3 = np.atleast_1d(
            np.round(hipx + PIXEL_OFFSET + EXTENT_MULT * sd).astype('int'))

        bp = np.r_[np.arange(y0[0], y1[0] + 1),
                   np.arange(y2[0], y3[0] + 1)]

        assert_equal(bp, background_pixels[0])

    def test_basename_datafile(self):
        # check that the right basename is returned
        pth = 'a/b/c.nx.hdf'
        assert_(basename_datafile(pth) == 'c')

        pth = 'c.nx.hdf'
        assert_(basename_datafile(pth) == 'c')

    def test_floodfield_correction(self):
        # check that flood field calculation works
        # the values were worked out by hand on a randomly
        # generated array
        test_norm = np.array([1.12290503, 1.23743017, 0.8603352,
                              0.70111732, 1.07821229])
        test_norm_sd = np.array([0.05600541, 0.05879208, 0.04902215,
                                 0.04425413, 0.05487956])

        fname = os.path.join(self.pth, 'flood.h5')
        with h5py.File(fname, 'r') as f:
            norm, norm_sd = create_detector_norm(f, 3.5, -3.5)

            assert_almost_equal(norm, test_norm, 6)
            assert_almost_equal(norm_sd, test_norm_sd, 6)

            norm, norm_sd = create_detector_norm(f, -3.5, 3.5)
            assert_almost_equal(norm, test_norm, 6)
            assert_almost_equal(norm_sd, test_norm_sd, 6)
