from __future__ import division
import numpy as np
import h5py
from . import peak_utils as ut
import refnx.util.general as general
import refnx.util.ErrorProp as EP
from . import parabolic_motion as pm
from . import event, rebin
from scipy.optimize import leastsq, curve_fit
from scipy.stats import t
import os
import os.path
import argparse
import re
import shutil
from time import gmtime, strftime
import string
import warnings
import io
import pandas as pd


# detector y pixel spacing in mm per pixel
Y_PIXEL_SPACING = 1.177

disc_openings = (60., 10., 25., 60.)
O_C1d, O_C2d, O_C3d, O_C4d = disc_openings
O_C1, O_C2, O_C3, O_C4 = np.radians(disc_openings)

DISCRADIUS = 350.
EXTENT_MULT = 2
PIXEL_OFFSET = 1


def catalogue(start, stop, path=None):
    """
    Extract interesting information from Platypus NeXUS files.

    Parameters
    ----------
    start : int
        start cataloguing from this run number.
    stop : int
        stop cataloguing at this run number
    path : str, optional
        path specifying location of NeXUS files

    Returns
    -------
    catalog : pd.DataFrame
        Dataframe containing interesting parameters from Platypus Nexus files
    """
    info = ['filename', 'end_time', 'sample_name', 'ss1vg', 'ss2vg', 'ss3vg',
            'ss4vg', 'omega', 'twotheta', 'bm1_counts', 'time', 'daq_dirname',
            'start_time']
    run_number = []
    d = {key:[] for key in info}

    if path is None:
        path = '.'

    for i in range(start, stop + 1):
        try:
            pn = PlatypusNexus(os.path.join(path, number_datafile(i)))
        except OSError:
            continue

        cat = pn.cat.cat
        run_number.append(i)

        for key, val in d.items():
            data = cat[key]
            if np.size(data) > 1 or type(data) is np.ndarray:
                data = data[0]
            if type(data) is bytes:
                data = data.decode()

            d[key].append(data)

    df = pd.DataFrame(d, index=run_number, columns=info)

    return df


class Catalogue(object):
    def __init__(self, h5data):
        d = {}
        file_path = os.path.realpath(h5data.filename)
        d['path'] = os.path.dirname(file_path)
        d['filename'] = h5data.filename
        d['end_time'] = h5data['entry1/end_time'][0]

        try:
            d['start_time'] = h5data['entry1/instrument/detector/start_time'][:]
        except KeyError:
            # start times don't exist in this file
            d['start_time'] = None

        d['sample_name'] = h5data['entry1/sample/name'][:]
        d['ss1vg'] = h5data['entry1/instrument/slits/first/vertical/gap'][:]
        d['ss2vg'] = h5data['entry1/instrument/slits/second/vertical/gap'][:]
        d['ss3vg'] = h5data['entry1/instrument/slits/third/vertical/gap'][:]
        d['ss4vg'] = h5data['entry1/instrument/slits/fourth/vertical/gap'][:]
        d['ss1hg'] = h5data['entry1/instrument/slits/first/horizontal/gap'][:]
        d['ss2hg'] = h5data['entry1/instrument/slits/second/horizontal/gap'][:]
        d['ss3hg'] = h5data['entry1/instrument/slits/third/horizontal/gap'][:]
        d['ss4hg'] = h5data['entry1/instrument/slits/fourth/horizontal/gap'][:]

        d['omega'] = h5data['entry1/instrument/parameters/omega'][:]
        d['twotheta'] = h5data['entry1/instrument/parameters/twotheta'][:]

        d['detector'] = h5data['entry1/data/hmm'][:]
        d['sth'] = h5data['entry1/sample/sth'][:]
        d['bm1_counts'] = h5data['entry1/monitor/bm1_counts'][:]
        d['total_counts'] = h5data['entry1/instrument/detector/total_counts'][:]
        d['time'] = h5data['entry1/instrument/detector/time'][:]
        d['mode'] = h5data['entry1/instrument/parameters/mode'][0].decode()

        try:
            event_directory_name = h5data[
                'entry1/instrument/detector/daq_dirname'][0]
            d['daq_dirname'] = event_directory_name.decode()
        except KeyError:
            # daq_dirname doesn't exist in this file
            d['daq_dirname'] = None

        d['t_bins'] = h5data['entry1/data/time_of_flight'][:].astype('float64')
        d['x_bins'] = h5data['entry1/data/x_bin'][:]
        d['y_bins'] = h5data['entry1/data/y_bin'][:]

        master, slave, frequency, phase = self._chopper_values(h5data)
        d['master'] = master
        d['slave'] = slave
        d['frequency'] = frequency
        d['phase'] = phase
        d['chopper2_distance'] = h5data[
            'entry1/instrument/parameters/chopper2_distance'][:]
        d['chopper3_distance'] = h5data[
            'entry1/instrument/parameters/chopper3_distance'][:]
        d['chopper4_distance'] = h5data[
            'entry1/instrument/parameters/chopper4_distance'][:]
        d['chopper1_phase_offset'] = h5data[
            'entry1/instrument/parameters/chopper1_phase_offset'][:]
        d['chopper2_phase_offset'] = h5data[
            'entry1/instrument/parameters/chopper2_phase_offset'][:]
        d['chopper3_phase_offset'] = h5data[
            'entry1/instrument/parameters/chopper3_phase_offset'][:]
        d['chopper4_phase_offset'] = h5data[
            'entry1/instrument/parameters/chopper4_phase_offset'][:]
        d['guide1_distance'] = h5data[
            'entry1/instrument/parameters/guide1_distance'][:]
        d['guide2_distance'] = h5data[
            'entry1/instrument/parameters/guide2_distance'][:]
        d['sample_distance'] = h5data[
            'entry1/instrument/parameters/sample_distance'][:]
        d['slit2_distance'] = h5data[
            'entry1/instrument/parameters/slit2_distance'][:]
        d['slit3_distance'] = h5data[
            'entry1/instrument/parameters/slit3_distance'][:]
        d['collimation_distance'] = d['slit3_distance'] - d['slit2_distance']
        d['dy'] = h5data[
            'entry1/instrument/detector/longitudinal_translation'][:]
        d['dz'] = h5data[
            'entry1/instrument/detector/vertical_translation'][:]
        d['original_file_name'] = h5data['entry1/experiment/file_name']
        # TODO put HDF file y pixel spacing in here.
        self.cat = d

    def __getattr__(self, item):
        return self.cat[item]

    @property
    def datafile_number(self):
        return datafile_number(self.filename)

    def _chopper_values(self, h5data):
        """
        Obtains chopper settings from NeXUS file

        Parameters
        ----------
        h5data : HDF5 NeXUS file
            datafile,

        Returns
        -------
        master, slave, frequency, phase : float, float, float, float
        """
        chopper1_speed = h5data['entry1/instrument/disk_chopper/ch1speed']
        chopper2_speed = h5data['entry1/instrument/disk_chopper/ch2speed']
        chopper3_speed = h5data['entry1/instrument/disk_chopper/ch3speed']
        chopper4_speed = h5data['entry1/instrument/disk_chopper/ch4speed']
        ch2phase = h5data['entry1/instrument/disk_chopper/ch2phase']
        ch3phase = h5data['entry1/instrument/disk_chopper/ch3phase']
        ch4phase = h5data['entry1/instrument/disk_chopper/ch4phase']

        if ('entry1/instrument/parameters/slave' in h5data and
                'entry1/instrument/parameters/master' in h5data):
            master = h5data['entry1/instrument/parameters/master'][0]
            slave = h5data['entry1/instrument/parameters/slave'][0]
        else:
            master = 1
            if abs(chopper2_speed[0]) > 10:
                slave = 2
            elif abs(chopper3_speed[0]) > 10:
                slave = 3
            else:
                slave = 4

        speeds = np.array([chopper1_speed,
                           chopper2_speed,
                           chopper3_speed,
                           chopper4_speed])

        phases = np.array([np.zeros_like(ch2phase),
                           ch2phase,
                           ch3phase,
                           ch4phase])

        return master, slave, speeds[0] / 60., phases[slave - 1]


def number_datafile(run_number):
    """
    From a run number figure out what the file name is.
    """
    return 'PLP{0:07d}.nx.hdf'.format(int(abs(run_number)))


def datafile_number(fname):
    """
    From a filename figure out what the run number was
    """
    regex = re.compile(".*PLP([0-9]{7}).nx.hdf")
    _fname = os.path.basename(fname)
    r = regex.search(_fname)

    if r:
        return int(r.groups()[0])

    return None


class PlatypusNexus(object):
    """
    Processes Platypus NeXus files to produce an intensity vs wavelength
    spectrum

    Parameters
    ----------
    h5data : HDF5 NeXus file or str
        An HDF5 NeXus file for Platypus, or a `str` containing the path
        to one
    """

    def __init__(self, h5data):
        """
        Initialises the PlatypusNexus object.
        """
        if type(h5data) == h5py.File:
            self.cat = Catalogue(h5data)
        else:
            with h5py.File(h5data, 'r') as h5data:
                self.cat = Catalogue(h5data)

        self.processed_spectrum = dict()

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        elif item in self.processed_spectrum:
            return self.processed_spectrum[item]
        else:
            raise AttributeError

    def process(self, h5norm=None, lo_wavelength=2.5, hi_wavelength=19.,
                background=True, direct=False, omega=None, twotheta=None,
                rebin_percent=1., wavelength_bins=None, normalise=True,
                integrate=-1, eventmode=None, event_folder=None, peak_pos=None,
                background_mask=None, normalise_bins=True, **kwds):
        r"""
        Processes the ProcessNexus object to produce a time of flight spectrum.
        The processed spectrum is stored in the `processed_spectrum` attribute.
        The specular spectrum is also returned from this function.

        Parameters
        ----------
        h5norm : HDF5 NeXus file
            The hdf5 file containing the floodfield data.
        lo_wavelength : float
            The low wavelength cutoff for the rebinned data (A).
        hi_wavelength : float
            The high wavelength cutoff for the rebinned data (A).
        background : bool
            Should a background subtraction be carried out?
        direct : bool
            Is it a direct beam you measured? This is so a gravity correction
            can be applied.
        omega : float
            Expected angle of incidence of beam. If this is None, then the
            rough angle of incidence is obtained from the NeXus file.
        twotheta : float
            Expected two theta value of specular beam. If this is None then
            the rough angle of incidence is obtained from the NeXus file.
        rebin_percent : float
            Specifies the rebinning percentage for the spectrum.  If
            `rebin_percent is None`, then no rebinning is done.
        wavelength_bins : array_like
            The wavelength bins for rebinning.  If `wavelength_bins is not
            None` then the `rebin_percent` parameter is ignored.
        normalise : bool
            Normalise by the monitor counts.
        integrate : int

            - integrate == -1
              the spectrum is integrated over all the scanpoints.
            - integrate >= 0
              the individual spectra are calculated individually.
              If `eventmode is not None` then integrate specifies which
              scanpoint to examine.

        eventmode : None or array_like
            If eventmode is `None` then the integrated detector image is used.
            If eventmode is an array then the array specifies the integration
            times (in seconds) for the detector image, e.g. [0, 20, 30] would
            result in two spectra. The first would contain data for 0 s to 20s,
            the second would contain data for 20 s to 30 s.  This option can
            only be used when `integrate >= -1`.
            If eventmode has zero length (e.g. []), then a single time interval
            for the entire acquisition is used, [0, acquisition_time].  This
            would source the image from the eventmode file, rather than the
            NeXUS file. Note: the two approaches will probably not give
            identical results, because the eventmode method adjusts the total
            acquisition time and beam monitor counts to the frame number of the
            last event detected (which may be quite different if the count rate
            is very low).
        event_folder : None or str
            Specifies the path for the eventmode data. If `event_folder is None`
            then the eventmode data is assumed to reside in the same directory
            as the NeXUS file. If event_folder is a string, then the string
            specifies the path to the eventmode data.
        peak_pos : None or (float, float)
            Specifies the peak position and peak standard deviation to use.
        background_mask : array_like
            An array of bool that specifies which y-pixels to use for
            background subtraction.  Should be the same length as the number of
            y pixels in the detector image.  Otherwise an automatic mask is
            applied (if background is True).
        normalise_bins : bool
            Divides the intensity in each wavelength bin by the width of the
            bin. This allows one to compare spectra even if they were processed
            with different rebin percentages.

        Notes
        -----
        After processing this object contains the following the following
        attributes:

        - path - path to the data file
        - datafilename - name of the datafile
        - datafile_number - datafile number.
        - m_topandtail - the corrected 2D detector image, (n_spectra, TOF, Y)
        - m_topandtail_sd - corresponding standard deviations
        - n_spectra - number of spectra in processed data
        - bm1_counts - beam montor counts, (n_spectra,)
        - m_spec - specular intensity, (n_spectra, TOF)
        - m_spec_sd - corresponding standard deviations
        - m_beampos - beam_centre for each spectrum, (n_spectra, )
        - m_lambda - wavelengths for each spectrum, (n_spectra, TOF)
        - m_lambda_fwhm - corresponding FWHM of wavelength distribution
        - m_lambda_hist - wavelength bins for each spectrum, (n_spectra, TOF)
        - m_spec_tof - TOF for each wavelength bin, (n_spectra, TOF)
        - mode - the Platypus mode, e.g. FOC/MT/POL/POLANAL/SB/DB
        - detector_z - detector height, (n_spectra, )
        - detector_y - sample-detector distance, (n_spectra, )
        - domega - collimation uncertainty
        - lopx - lowest extent of specular beam (in y pixels), (n_spectra, )
        - hipx - highest extent of specular beam (in y pixels), (n_spectra, )

        Returns
        -------
        m_lambda, m_spec, m_spec_sd: np.ndarray
            Arrays containing the wavelength, specular intensity as a function
            of wavelength, standard deviation of specular intensity
        """
        cat = self.cat

        scanpoint = 0

        # beam monitor counts for normalising data
        bm1_counts = cat.bm1_counts.astype('float64')

        # TOF bins
        TOF = cat.t_bins.astype('float64')

        # This section controls how multiple detector images are handled.
        # We want event streaming.
        if eventmode is not None:
            scanpoint = integrate
            if integrate == -1:
                scanpoint = 0

            output = self.process_event_stream(scanpoint=scanpoint,
                                               frame_bins=eventmode,
                                               event_folder=event_folder)
            frame_bins, detector, bm1_counts = output

            start_time = np.zeros(np.size(detector, 0))
            if cat.start_time is not None:
                start_time += cat.start_time[scanpoint]
                start_time += frame_bins[:-1]
        else:
            # we don't want detector streaming
            detector = cat.detector
            scanpoint = 0

            # integrate over all spectra
            if integrate == -1:
                detector = np.sum(detector, 0)[np.newaxis, ]
                bm1_counts[:] = np.sum(bm1_counts)

            start_time = np.zeros(np.size(detector, 0))
            if cat.start_time is not None:
                for idx in range(start_time.size):
                    start_time[idx] = cat.start_time[idx]

        n_spectra = np.size(detector, 0)

        # Up until this point detector.shape=(N, T, Y)
        # pre-average over x, leaving (n, t, y) also convert to dp
        detector = np.sum(detector, axis=3, dtype='float64')

        # detector shape should now be (n, t, y)
        # calculate the counting uncertainties
        detector_sd = np.sqrt(detector)
        bm1_counts_sd = np.sqrt(bm1_counts)

        # detector normalisation with a water file
        if h5norm:
            x_bins = cat.x_bins[scanpoint]
            # shape (y,)
            detector_norm, detector_norm_sd = create_detector_norm(h5norm,
                                                                   x_bins[0],
                                                                   x_bins[1])
            # detector has shape (N, T, Y), shape of detector_norm should
            # broadcast to (1, 1, y)
            # TODO: Correlated Uncertainties?
            detector, detector_sd = EP.EPdiv(detector, detector_sd,
                                             detector_norm, detector_norm_sd)

        # shape of these is (n_spectra, TOFbins)
        m_spec_tof_hist = np.zeros((n_spectra, np.size(TOF, 0)),
                                   dtype='float64')
        m_lambda_hist = np.zeros((n_spectra, np.size(TOF, 0)),
                                 dtype='float64')
        m_spec_tof_hist[:] = TOF

        """
        chopper to detector distances
        note that if eventmode is specified the n_spectra is NOT
        equal to the number of entries in e.g. /longitudinal_translation
        this means you have to copy values in from the correct scanpoint
        """
        flight_distance = np.zeros(n_spectra, dtype='float64')
        d_cx = np.zeros(n_spectra, dtype='float64')
        detpositions = np.zeros(n_spectra, dtype='float64')

        # The angular divergence of the instrument
        domega = np.zeros(n_spectra, dtype='float64')

        phase_angle = np.zeros(n_spectra, dtype='float64')

        # process each of the spectra taken in the detector image
        originalscanpoint = scanpoint
        for idx in range(n_spectra):
            freq = cat.frequency[scanpoint]

            # calculate the angular divergence
            domega[idx] = general.div(cat.ss2vg[scanpoint],
                                      cat.ss3vg[scanpoint],
                                      (cat.slit3_distance[0]
                                       - cat.slit2_distance[0]))[0]

            """
            work out the total flight length
            IMPORTANT: this varies as a function of twotheta. This is
            because the Platypus detector does not move on an arc.
            At high angles chod can be ~ 0.75% different. This is will
            visibly shift fringes.
            """
            if omega is None:
                omega = cat.omega[scanpoint]
            if twotheta is None:
                twotheta = cat.twotheta[scanpoint]
            output = self.chod(omega, twotheta, scanpoint=scanpoint)
            flight_distance[idx], d_cx[idx] = output

            # calculate phase openings
            output = self.phase_angle(scanpoint)
            phase_angle[scanpoint], master_opening = output

            """
            toffset - the time difference between the magnet pickup on the
            choppers (TTL pulse), which is situated in the middle of the
            chopper window, and the trailing edge of chopper 1, which is
            supposed to be time0.  However, if there is a phase opening this
            time offset has to be relocated slightly, as time0 is not at the
            trailing edge.
            """
            poff = cat.chopper1_phase_offset[0]
            poffset = 1.e6 * poff / (2. * 360. * freq)
            toffset = (poffset
                       + 1.e6 * master_opening / 2 / (2 * np.pi) / freq
                       - 1.e6 * phase_angle[scanpoint] / (360 * 2 * freq))
            m_spec_tof_hist[idx] -= toffset

            detpositions[idx] = cat.dy[scanpoint]

            if eventmode is not None:
                m_spec_tof_hist[:] = TOF - toffset
                flight_distance[:] = flight_distance[0]
                detpositions[:] = detpositions[0]
                domega[:] = domega[0]
                d_cx[:] = d_cx[0]
                phase_angle[:] = phase_angle[0]
                break
            else:
                scanpoint += 1

        scanpoint = originalscanpoint

        # convert TOF to lambda
        # m_spec_tof_hist (n, t) and chod is (n,)
        m_lambda_hist = general.velocity_wavelength(
                    1.e3 * flight_distance[:, np.newaxis] / m_spec_tof_hist)

        m_lambda = 0.5 * (m_lambda_hist[:, 1:] + m_lambda_hist[:, :-1])
        TOF -= toffset

        # gravity correction if direct beam
        if direct:
            # TODO: Correlated Uncertainties?
            output = correct_for_gravity(detector,
                                         detector_sd,
                                         m_lambda,
                                         self.cat.collimation_distance,
                                         self.cat.dy,
                                         lo_wavelength,
                                         hi_wavelength)
            detector, detector_sd, m_gravcorrcoefs = output
            beam_centre, beam_sd = find_specular_ridge(detector, detector_sd)
            # beam_centre = m_gravcorrcoefs
        else:
            beam_centre, beam_sd = find_specular_ridge(detector, detector_sd)

        # you want to specify the specular ridge on the averaged detector image
        if peak_pos is not None:
            beam_centre = np.ones(n_spectra) * peak_pos[0]
            beam_sd = np.ones(n_spectra) * peak_pos[1]

        '''
        Rebinning in lambda for all detector
        Rebinning is the default option, but sometimes you don't want to.
        detector shape input is (n, t, y)
        '''
        if wavelength_bins is not None:
            rebinning = wavelength_bins
        elif 0. < rebin_percent < 15.:
            rebinning = calculate_wavelength_bins(lo_wavelength,
                                                  hi_wavelength,
                                                  rebin_percent)

        # rebin_percent percentage is zero. No rebinning, just cutoff
        # wavelength
        else:
            rebinning = m_lambda_hist[0, :]
            rebinning = rebinning[np.searchsorted(rebinning, lo_wavelength):
                                  np.searchsorted(rebinning, hi_wavelength)]

        """
        now do the rebinning for all the N detector images
        rebin.rebinND could do all of these at once.  However, m_lambda_hist
        could vary across the range of spectra.  If it was the same I could
        eliminate the loop.
        """
        output = []
        output_sd = []
        for idx in range(n_spectra):
            # TODO: Correlated Uncertainties?
            plane, plane_sd = rebin.rebin_along_axis(detector[idx],
                                                     m_lambda_hist[idx],
                                                     rebinning,
                                                     y1_sd=detector_sd[idx])
            output.append(plane)
            output_sd.append(plane_sd)

        detector = np.array(output)
        detector_sd = np.array(output_sd)

        if len(detector.shape) == 2:
            detector = detector[np.newaxis, ]
            detector_sd = detector_sd[np.newaxis, ]

        # (1, T)
        m_lambda_hist = np.atleast_2d(rebinning)

        """
        Divide the detector intensities by the width of the wavelength bin.
        This is so the intensities between different rebinning strategies can
        be compared.
        """
        if normalise_bins:
            div = 1 / np.ediff1d(m_lambda_hist[0])[:, np.newaxis]
            detector, detector_sd = EP.EPmulk(detector,
                                              detector_sd,
                                              div)

        # convert the wavelength base to a timebase
        m_spec_tof_hist = (0.001 * flight_distance[:, np.newaxis]
                           / general.wavelength_velocity(m_lambda_hist))

        m_lambda = 0.5 * (m_lambda_hist[:, 1:] + m_lambda_hist[:, :-1])

        m_spec_tof = (0.001 * flight_distance[:, np.newaxis]
                      / general.wavelength_velocity(m_lambda))

        # we want to integrate over the following pixel region
        lopx = np.floor(beam_centre - beam_sd * EXTENT_MULT).astype('int')
        hipx = np.ceil(beam_centre + beam_sd * EXTENT_MULT).astype('int')

        m_spec = np.zeros((n_spectra, np.size(detector, 1)))
        m_spec_sd = np.zeros_like(m_spec)

        # background subtraction
        if background:
            if background_mask is not None:
                # background_mask is (Y), need to make 3 dimensional (N, T, Y)
                # first make into (T, Y)
                backgnd_mask = np.repeat(background_mask[np.newaxis, :],
                                         detector.shape[1],
                                         axis=0)
                # make into (N, T, Y)
                full_backgnd_mask = np.repeat(backgnd_mask[np.newaxis, :],
                                              n_spectra,
                                              axis=0)
            else:
                # there may be different background regions for each spectrum
                # in the file
                y1 = np.round(lopx - PIXEL_OFFSET).astype('int')
                y0 = np.round(y1 - (EXTENT_MULT * beam_sd)).astype('int')

                y2 = np.round(hipx + PIXEL_OFFSET).astype('int')
                y3 = np.round(y2 + (EXTENT_MULT * beam_sd)).astype('int')

                full_backgnd_mask = np.zeros_like(detector, dtype='bool')
                for i in range(n_spectra):
                    full_backgnd_mask[i, :, y0[i]: y1[i]] = True
                    full_backgnd_mask[i, :, y2[i] + 1: y3[i] + 1] = True

            # TODO: Correlated Uncertainties?
            detector, detector_sd = background_subtract(detector,
                                                        detector_sd,
                                                        full_backgnd_mask)

        '''
        top and tail the specular beam with the known beam centres.
        All this does is produce a specular intensity with shape (N, T),
        i.e. integrate over specular beam
        '''
        for i in range(n_spectra):
            m_spec[i] = np.sum(detector[i, :, lopx[i]: hipx[i] + 1], axis=1)
            sd = np.sum(detector_sd[i, :, lopx[i]: hipx[i] + 1] ** 2,
                        axis=1)
            m_spec_sd[i] = np.sqrt(sd)

        # assert np.isfinite(m_spec).all()
        # assert np.isfinite(m_specSD).all()
        # assert np.isfinite(detector).all()
        # assert np.isfinite(detectorSD).all()

        # normalise by beam monitor 1.
        if normalise:
            m_spec, m_spec_sd = EP.EPdiv(m_spec,
                                         m_spec_sd,
                                         bm1_counts[:, np.newaxis],
                                         bm1_counts_sd[:, np.newaxis])

            output = EP.EPdiv(detector,
                              detector_sd,
                              bm1_counts[:, np.newaxis, np.newaxis],
                              bm1_counts_sd[:, np.newaxis, np.newaxis])
            detector, detector_sd = output

        '''
        now work out dlambda/lambda, the resolution contribution from
        wavelength.
        van Well, Physica B,  357(2005) pp204-207), eqn 4.
        this is only an approximation for our instrument, as the 2nd and 3rd
        discs have smaller openings compared to the master chopper.
        Therefore the burst time needs to be looked at.
        '''
        tau_da = m_spec_tof_hist[:, 1:] - m_spec_tof_hist[:, :-1]

        m_lambda_fwhm = general.resolution_double_chopper(m_lambda,
                                     z0=d_cx[:, np.newaxis] / 1000.,
                                     freq=cat.frequency[:, np.newaxis],
                                     L=flight_distance[:, np.newaxis] / 1000.,
                                     H=cat.ss2vg[originalscanpoint] / 1000.,
                                     xsi=phase_angle[:, np.newaxis],
                                     tau_da=tau_da)

        m_lambda_fwhm *= m_lambda

        # put the detector positions and mode into the dictionary as well.
        detector_z = cat.dz
        detector_y = cat.dy
        mode = cat.mode

        d = dict()
        d['path'] = cat.path
        d['datafilename'] = cat.filename
        d['datafile_number'] = cat.datafile_number

        if h5norm is not None:
            d['normfilename'] = h5norm.filename
        d['m_topandtail'] = detector
        d['m_topandtail_sd'] = detector_sd
        d['n_spectra'] = n_spectra
        d['bm1_counts'] = bm1_counts
        d['m_spec'] = m_spec
        d['m_spec_sd'] = m_spec_sd
        d['m_beampos'] = beam_centre
        d['m_lambda'] = m_lambda
        d['m_lambda_fwhm'] = m_lambda_fwhm
        d['m_lambda_hist'] = m_lambda_hist
        d['m_spec_tof'] = m_spec_tof
        d['mode'] = mode
        d['detector_z'] = detector_z
        d['detector_y'] = detector_y
        d['domega'] = domega
        d['lopx'] = lopx
        d['hipx'] = hipx
        d['start_time'] = start_time

        self.processed_spectrum = d
        return m_lambda, m_spec, m_spec_sd

    def phase_angle(self, scanpoint=0):
        """
        Calculates the phase angle for a given scanpoint

        Parameters
        ----------
        scanpoint : int
            The scanpoint you're interested in

        Returns
        -------
        phase_angle, master_opening : float
            The phase angle in degrees, and the angular opening of the master
            chopper
        """
        cat = self.cat
        master = cat.master
        slave = cat.slave
        disc_phase = cat.phase[scanpoint]
        phase_angle = 0

        if master == 1:
            phase_angle += 0.5 * O_C1d
            master_opening = O_C1
        elif master == 2:
            phase_angle += 0.5 * O_C2d
            master_opening = O_C2
        elif master == 3:
            phase_angle += 0.5 * O_C3d
            master_opening = O_C3

        if slave == 2:
            phase_angle += 0.5 * O_C2d
            phase_angle += -disc_phase - cat.chopper2_phase_offset[0]
        elif slave == 3:
            phase_angle += 0.5 * O_C3d
            phase_angle += -disc_phase - cat.chopper3_phase_offset[0]
        elif slave == 4:
            phase_angle += 0.5 * O_C4d
            phase_angle += disc_phase - cat.chopper4_phase_offset[0]

        return phase_angle, master_opening

    def chod(self, omega=0., twotheta=0., scanpoint=0):
        """
        Calculates the flight length of the neutrons in the Platypus instrument.

        Parameters
        ----------
        omega : float, optional
            Rough angle of incidence
        twotheta : float, optional
            Rough 2 theta angle
        scanpoint : int, optional
            Which dataset is being considered

        Returns
        -------
        chod, d_cx : float, float
            Flight distance (mm), distance between chopper discs (mm)
        """

        chod = 0

        # guide 1 is the single deflection mirror (SB)
        # its distance is from chopper 1 to the middle of the mirror (1m long)
        # guide 2 is the double deflection mirror (DB)
        # its distance is from chopper 1 to the middle of the second of the
        # compound mirrors! (a bit weird, I know).

        cat = self.cat
        mode = cat.mode

        # Find out chopper pairing
        master = cat.master
        slave = cat.slave

        d_cx = 0

        if master == 1:
            chod = 0
        elif master == 2:
            chod -= cat.chopper2_distance[0]
            d_cx -= chod
        elif master == 3:
            chod -= cat.chopper3_distance[0]
            d_cx -= chod
        else:
            raise ValueError("Chopper pairing should be one of '12', '13',"
                             "'14', '23', '24', '34'")

        if slave == 2:
            chod -= cat.chopper2_distance[0]
            d_cx += cat.chopper2_distance[0]
        elif slave == 3:
            chod -= cat.chopper3_distance[0]
            d_cx += cat.chopper3_distance[0]
        elif slave == 4:
            chod -= cat.chopper4_distance[0]
            d_cx += cat.chopper4_distance[0]

        # start of flight length is midway between master and slave, but master
        # may not necessarily be disk 1. However, all instrument lengths are
        # measured from disk1
        chod /= 2.

        if mode in ['FOC', 'POL', 'MT', 'POLANAL']:
            chod += cat.sample_distance[0]
            chod += cat.dy[scanpoint] / np.cos(np.radians(twotheta))

        elif mode == 'SB':
            # assumes guide1_distance is in the MIDDLE OF THE MIRROR
            chod += cat.guide1_distance[0]
            chod += ((cat.sample_distance[0] - cat.guide1_distance[0])
                     / np.cos(np.radians(omega)))
            if twotheta > omega:
                chod += (cat.dy[scanpoint] /
                         np.cos(np.radians(twotheta - omega)))
            else:
                chod += (cat.dy[scanpoint]
                         / np.cos(np.radians(omega - twotheta)))

        elif mode == 'DB':
            # guide2_distance in in the middle of the 2nd compound mirror
            # guide2_distance - longitudinal length from midpoint1 -> midpoint2
            #  + direct length from midpoint1->midpoint2
            chod += (cat.guide2_distance[0]
                     + 600. * np.cos(np.radians(1.2))
                     * (1 - np.cos(np.radians(2.4))))

            # add on distance from midpoint2 to sample
            chod += ((cat.sample_distance[0] - cat.guide2_distance[0])
                     / np.cos(np.radians(4.8)))

            # add on sample -> detector
            if twotheta > omega:
                chod += (cat.dy[scanpoint]
                         / np.cos(np.radians(twotheta - 4.8)))
            else:
                chod += (cat.dy[scanpoint]
                         / np.cos(np.radians(4.8 - twotheta)))

        return chod, d_cx

    def process_event_stream(self, t_bins=None, x_bins=None, y_bins=None,
                             frame_bins=None, scanpoint=0, event_folder=None):
        """
        Processes the event mode dataset for the NeXUS file. Assumes that
        there is a event mode directory in the same directory as the NeXUS
        file, as specified by in 'entry1/instrument/detector/daq_dirname'

        Parameters
        ----------
        frame_bins : array_like, optional
            specifies the frame bins required in the image. If
            framebins = [5, 10, 120] you will get 2 images.  The first starts
            at 5s and finishes at 10s. The second starts at 10s and finishes
            at 120s. If frame_bins has zero length, e.g. [], then a single
            interval consisting of the entire acquisition time is used:
            [0, acquisition_time].
        t_bins : array_like, optional
            specifies the time bins required in the image
        x_bins : array_like, optional
            specifies the x bins required in the image
        y_bins : array_like, optional
            specifies the y bins required in the image
        scanpoint : int, optional
            Scanpoint you are interested in
        event_folder : None or str
            Specifies the path for the eventmode data. If `event_folder is None`
            then the eventmode data is assumed to reside in the same directory
            as the NeXUS file. If event_folder is a string, then the string
            specifies the path to the eventmode data.

        Returns
        -------
        frame_bins, detector, bm1_counts

        Create a new detector image based on the t_bins, x_bins, y_bins and
        frame_bins you supply to the method (these should all be lists/numpy
        arrays specifying the edges of the required bins). If these are not
        specified, then the default bins are taken from the nexus file. This
        would essentially return the same detector image as the nexus file.
        However, you can specify the frame_bins list to generate detector
        images based on subdivided periods of the total acquisition.
        For example if framebins = [5, 10, 120] you will get 2 images.  The
        first starts at 5s and finishes at 10s. The second starts at 10s
        and finishes at 120s. The frame_bins are clipped to the total
        acquisition time if necessary.
        """
        cat = self.cat

        if not t_bins:
            t_bins = cat.t_bins
        if not y_bins:
            y_bins = cat.y_bins
        if not x_bins:
            x_bins = cat.x_bins
        if frame_bins is None or np.size(frame_bins) == 0:
            frame_bins = [0, cat.time[scanpoint]]

        total_acquisition_time = cat.time[scanpoint]
        frequency = cat.frequency[scanpoint]

        bm1_counts_for_scanpoint = cat.bm1_counts[scanpoint]

        event_directory_name = cat.daq_dirname

        _eventpath = cat.path
        if event_folder is not None:
            _eventpath = event_folder

        stream_filename = os.path.join(_eventpath,
                                       event_directory_name,
                                       'DATASET_%d' % scanpoint,
                                       'EOS.bin')

        with io.open(stream_filename, 'rb') as f:
            events, end_of_last_event = event.events(f,
                                    max_frames=int(frame_bins[-1] * frequency))

        output = event.process_event_stream(events,
                                            np.asfarray(frame_bins)
                                            * frequency,
                                            t_bins,
                                            y_bins,
                                            x_bins)

        detector, new_frame_bins = output

        new_frame_bins /= frequency

        bm1_counts = new_frame_bins[1:] - new_frame_bins[:-1]
        bm1_counts *= (bm1_counts_for_scanpoint / total_acquisition_time)

        return new_frame_bins, detector, bm1_counts

    def write_spectrum_dat(self, f, scanpoint=0):
        """
        This method writes a dat representation of the corrected spectrum to
        file.

        Parameters
        ----------
        f : file-like or str
            The file to write the spectrum to, or a str that specifies the file
            name
        scanpoint : int
            Which scanpoint to write
        """
        if self.processed_spectrum is None:
            return False

        m_lambda = self.processed_spectrum['m_lambda'][scanpoint]
        m_spec = self.processed_spectrum['m_spec'][scanpoint]
        m_spec_sd = self.processed_spectrum['m_spec_sd'][scanpoint]
        m_lambda_fwhm = self.processed_spectrum['m_lambda_fwhm'][scanpoint]

        stacked_data = np.c_[m_lambda, m_spec, m_spec_sd, m_lambda_fwhm]
        np.savetxt(f, stacked_data, delimiter='\t')

        return True

    def write_spectrum_xml(self, f, scanpoint=0):
        """
        This method writes an XML representation of the corrected spectrum to
        file.

        Parameters
        ----------
        f : file-like or str
            The file to write the spectrum to, or a str that specifies the file
            name
        scanpoint : int
            Which scanpoint to write
        """
        spectrum_template = """<?xml version="1.0"?>
        <REFroot xmlns="">
        <REFentry time="$time">
        <Title>$title</Title>
        <REFdata axes="lambda" rank="1" type="POINT" spin="UNPOLARISED" dim="$n_spectra">
        <Run filename="$runnumber"/>
        <R uncertainty="dR">$r</R>
        <lambda uncertainty="dlambda" units="1/A">$l</lambda>
        <dR type="SD">$dr</dR>
        <dlambda type="_FWHM" units="1/A">$dl</dlambda>
        </REFdata>
        </REFentry>
        </REFroot>"""
        if self.processed_spectrum is None:
            return

        s = string.Template(spectrum_template)
        d = dict()
        d['title'] = self.cat.sample_name
        d['time'] = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

        m_lambda = self.processed_spectrum['m_lambda']
        m_spec = self.processed_spectrum['m_spec']
        m_spec_sd = self.processed_spectrum['m_spec_sd']
        m_lambda_fwhm = self.processed_spectrum['m_lambda_fwhm']

        # sort the data
        sorted = np.argsort(self.m_lambda[0])

        r = m_spec[:, sorted]
        l = m_lambda[:, sorted]
        dl = m_lambda_fwhm [:, sorted]
        dr = m_spec_sd[:, sorted]
        d['n_spectra'] = self.processed_spectrum['n_spectra']
        d['runnumber'] = 'PLP{:07d}'.format(self.cat.datafile_number)

        d['r'] = repr(r[scanpoint].tolist()).strip(',[]')
        d['dr'] = repr(dr[scanpoint].tolist()).strip(',[]')
        d['l'] = repr(l[scanpoint].tolist()).strip(',[]')
        d['dl'] = repr(dl[scanpoint].tolist()).strip(',[]')
        thefile = s.safe_substitute(d)

        g = f
        auto_fh = None

        if not hasattr(f, 'write'):
            auto_fh = open(f, 'wb')
            g = auto_fh

        if 'b' in g.mode:
            thefile = thefile.encode('utf-8')

        g.write(thefile)
        g.truncate()

        if auto_fh is not None:
            auto_fh.close()

        return True

    @property
    def spectrum(self):
        return (self.processed_spectrum['m_lambda'],
                self.processed_spectrum['m_spec'],
                self.processed_spectrum['m_spec_sd'],
                self.processed_spectrum['m_lambda_fwhm'])


def create_detector_norm(h5norm, x_min, x_max):
    """
    Produces a detector normalisation array for Platypus.
    Here we average over N, T and X to provide  a relative efficiency for each
    y wire.

    Parameters
    ----------
    h5norm : hdf5 file
        Containing a flood field run (water)
    x_min : int
        Minimum x pixel to use
    x_max : int
        Maximum x pixel to use

    Returns
    -------
    norm, norm_sd : array_like
        1D array containing the normalisation data for each y pixel
    """
    # sum over N and T
    detector = h5norm['entry1/data/hmm']
    norm = np.sum(detector, axis=(0, 1),
                  dtype='float64')

    # By this point you have norm[y][x]
    norm = norm[:, x_min: x_max + 1]
    norm = np.sum(norm, axis=1)

    mean = np.mean(norm)

    return norm / mean, np.sqrt(norm) / mean


def background_subtract(detector, detector_sd, background_mask):
    """
    Background subtraction of Platypus detector image.
    Shape of detector is (N, T, Y), do a linear background subn for each
    (N, T) slice.

    Parameters
    ----------
    detector : np.ndarray
        detector array with shape (N, T, Y).
    detector_sd : np.ndarray
        standard deviations for detector array
    background_mask : array_like
        array of bool with shape (N, T, Y) that specifies which Y pixels to use
        for background subtraction.

    Returns
    -------
    detector, detector_sd : np.ndarray, np.ndarray
        Detector image with background subtracted
    """
    ret = np.zeros_like(detector)
    ret_sd = np.zeros_like(detector)

    for idx in np.ndindex(detector.shape[0: 2]):
        ret[idx], ret_sd[idx] = background_subtract_line(detector[idx],
                                                         detector_sd[idx],
                                                         background_mask[idx])
    return ret, ret_sd


def background_subtract_line(profile, profile_sd, background_mask):
    """
    Performs a linear background subtraction on a 1D peak profile

    Parameters
    ----------
    profile : np.ndarray
        1D profile
    profile_sd : np.ndarray
        standard deviations for profile
    background_mask : array_like
        array of bool that specifies which Y pixels to use for background
        subtraction.
    """

    # which values to use as a background region
    mask = np.array(background_mask).astype('bool')
    x_vals = np.where(mask)[0]

    try:
        y_vals = profile[x_vals]
    except IndexError:
        print(x_vals)

    y_sdvals = profile_sd[x_vals]
    x_vals = x_vals.astype('float')

    # some SD values may have 0 SD, which will screw up curvefitting.
    y_sdvals = np.where(y_sdvals == 0, 1, y_sdvals)

    # equation for a straight line
    def f(x, a, b):
        return a + b * x

    # estimate the linear fit
    y_bar = np.mean(y_vals)
    x_bar = np.mean(x_vals)
    bhat = np.sum((x_vals - x_bar) * (y_vals - y_bar))
    bhat /= np.sum((x_vals - x_bar) ** 2)
    ahat = y_bar - bhat * x_bar

    # get the weighted fit values
    # we know the absolute sigma values
    popt, pcov = curve_fit(f, x_vals, y_vals, sigma=y_sdvals,
                           p0=np.array([ahat, bhat]), absolute_sigma=True)

    def CI(xx, pcovmat):
        return (pcovmat[0, 0] + pcovmat[1, 0] * xx
                + pcovmat[0, 1] * xx + pcovmat[1, 1] * (xx ** 2))

    bkgd = f(np.arange(np.size(profile, 0)), popt[0], popt[1])

    # now work out confidence intervals
    # TODO, should this be confidence interval or prediction interval?
    # if you try to do a fit which has a singular matrix
    if np.isfinite(pcov).all():
        bkgd_sd = np.asarray([CI(x, pcov) for x in np.arange(len(profile))],
                             dtype='float64')
    else:
        bkgd_sd = np.zeros_like(bkgd)

    bkgd_sd = np.sqrt(bkgd_sd)

    # get the t value for a two sided student t test at the 68.3 confidence
    # level
    bkgd_sd *= t.isf(0.1585, np.size(x_vals, 0) - 2)

    return EP.EPsub(profile, profile_sd, bkgd, bkgd_sd)


def find_specular_ridge(detector, detector_sd, starting_offset=50,
                        tolerance=0.01):
    """
    Find the specular ridges in a detector(n, t, y) plot. Assumes that the
    specular ridge **does not** change position.

    Parameters
    ----------
    detector : array_like
        detector array
    detector_sd : array_like
        standard deviations of detector array

    Returns
    -------
    centre, SD:
        peak centres and standard deviations of peak width
    """
    beam_centre = np.zeros(np.size(detector, 0))
    beam_sd = np.zeros(np.size(detector, 0))
    search_increment = 50

    starting_offset = abs(starting_offset)

    n_increments = ((np.size(detector, 1) - starting_offset)
                    // search_increment)

    for j in range(np.size(detector, 0)):
        last_centre = -1.
        last_sd = -1.
        converged = False

        for i in range(n_increments):
            how_many = -starting_offset - search_increment * i

            det_subset = detector[j, -1: how_many: -1]
            det_sd_subset = detector_sd[j, -1: how_many: -1]

            # Uncertainties code takes a while to run
            # total_y = np.sum(det_subset, axis=0)
            y_cross = np.sum(det_subset, axis=0)
            y_cross_sd = np.sqrt(np.sum(det_sd_subset ** 2., axis=0))

            # find the centroid and gauss peak in the last sections of the TOF
            # plot

            try:
                centroid, gauss_peak = ut.peak_finder(y_cross, sigma=y_cross_sd)
            except RuntimeError:
                continue

            if (abs((gauss_peak[0] - last_centre) / last_centre) < tolerance
                and abs((gauss_peak[1] - last_sd) / last_sd) < tolerance):
                last_centre = gauss_peak[0]
                last_sd = gauss_peak[1]
                converged = True
                break

            last_centre = gauss_peak[0]
            last_sd = gauss_peak[1]

        if not converged:
            warnings.warn('specular ridge search did not work properly'
                          ' using last known centre', RuntimeWarning)

        beam_centre[j] = last_centre
        beam_sd[j] = np.abs(last_sd)

    return beam_centre, beam_sd


def correct_for_gravity(detector, detector_sd, lamda, coll_distance,
                        sample_det, lo_wavelength, hi_wavelength,
                        theta=0):
    """
    Returns a gravity corrected yt plot, given the data, its associated errors,
    the wavelength corresponding to each of the time bins, and the trajectory
    of the neutrons. Low lambda and high Lambda are wavelength cutoffs to
    ignore.

    Parameters
    ----------
    detector : np.ndarray
        Detector image. Has shape (N, T, Y)
    detector_sd : np.ndarray
        Standard deviations of detector image
    lamda : np.ndarray
        Wavelengths corresponding to the detector image, has shape (N, T)
    coll_distance : float
        Collimation distance between slits, mm
    sample_det : float
        Sample - detector distance, mm
    lo_wavelength : float
        Low wavelength cut off, Angstrom
    hi_wavelength : float
        High wavelength cutoff, Angstrom
    theta : float
        Angle between second collimation slit, first collimation slit, and
        horizontal

    Returns
    -------
    corrected_data, corrected_data_sd, m_gravcorrcoefs : np.ndarray, np.ndarray, np.ndarray
        Corrected image. This is a theoretical prediction where the spectral
        ridge is for each wavelength.  This will be used to calculate the
        actual angle of incidence in the reduction process.

    """
    num_lambda = np.size(lamda, axis=1)

    x_init = np.arange((np.size(detector, axis=2) + 1) * 1.) - 0.5

    m_gravcorrcoefs = np.zeros((np.size(detector, 0)), dtype='float64')

    corrected_data = np.zeros_like(detector)
    corrected_data_sd = np.zeros_like(detector)

    for spec in range(np.size(detector, 0)):
        neutron_speeds = general.wavelength_velocity(lamda[spec])
        trajectories = pm.find_trajectory(coll_distance / 1000., theta, neutron_speeds)
        travel_distance = (coll_distance + sample_det[spec]) / 1000.

        # centres(t,)
        # TODO, don't use centroids, use Gaussian peak
        centroids = np.apply_along_axis(ut.centroid, 1, detector[spec])
        lopx = np.searchsorted(lamda[spec], lo_wavelength)
        hipx = np.searchsorted(lamda[spec], hi_wavelength)

        def f(tru_centre):
            deflections = pm.y_deflection(trajectories[lopx: hipx], neutron_speeds[lopx: hipx], travel_distance)

            model = 1000. * deflections / Y_PIXEL_SPACING + tru_centre
            diff = model - centroids[lopx: hipx, 0]
            diff = diff[~np.isnan(diff)]
            return diff

        # find the beam centre for an infinitely fast neutron
        x0 = np.array([np.nanmean(centroids[lopx: hipx, 0])])
        res = leastsq(f, x0)
        m_gravcorrcoefs[spec] = res[0][0]

        total_deflection = 1000. * pm.y_deflection(trajectories, neutron_speeds, travel_distance)
        total_deflection /= Y_PIXEL_SPACING

        x_rebin = x_init.T + total_deflection[:, np.newaxis]
        for wavelength in range(np.size(detector, axis=1)):
            output = rebin.rebin(x_init,
                                 detector[spec, wavelength],
                                 x_rebin[wavelength],
                                 y1_sd=detector_sd[spec, wavelength])

            corrected_data[spec, wavelength] = output[0]
            corrected_data_sd[spec, wavelength] = output[1]

    return corrected_data, corrected_data_sd, m_gravcorrcoefs


def calculate_wavelength_bins(lo_wavelength, hi_wavelength, rebin_percent):
    """
    Calculates optimal logarithmically spaced wavelength histogram bins. The
    bins are equal size in log10 space, but they may not be exactly be
    `rebin_percent` in size. The limits would have to change slightly for that.

    Parameters
    ----------
    lo_wavelength : float
        Low wavelength cutoff
    hi_wavelength : float
        High wavelength cutoff
    rebin_percent : float
        Rebinning percentage

    Returns
    -------
    wavelength_bins : np.ndarray
    """
    frac = (rebin_percent / 100.) + 1
    lowspac = rebin_percent / 100. * lo_wavelength
    hispac = rebin_percent / 100. * hi_wavelength

    lowl = lo_wavelength - lowspac / 2.
    hil = hi_wavelength + hispac / 2.
    num_steps = int(np.floor(np.log10(hil / lowl) / np.log10(frac)) + 1)
    rebinning = np.logspace(np.log10(lowl), np.log10(hil), num=num_steps)
    return rebinning


def accumulate_HDF_files(files):
    r"""
    Accumulates HDF files together, writing an accumulated file in the current
    directory. The accumulated datafile is written in the current directory
    (os.getcwd()) and has a filename based on the first file, prepended by
    'ADD\_'. For example, if the first file is PLP0000708.nx.hdf then the
    accumulated file is ADD_PLP0000708.nx.hdf.

    Parameters
    ----------
    files : list
        Strings specifying NeXUS filenames to be added together.

    """
    # don't do anything if no files were supplied.
    if not len(files):
        return None

    # the first file is the "master file", lets copy it.
    file = files[0]

    pth = _check_HDF_file(file)
    if not pth:
        raise ValueError('All files must refer to an hdf5 file')

    new_name = 'ADD_' + os.path.basename(pth)

    shutil.copy(pth,
                os.path.join(os.getcwd(), new_name))

    master_file = os.path.join(os.getcwd(), new_name)
    with h5py.File(master_file, 'r+') as h5master:
        # now go through each file and accumulate numbers:
        for file in files[1:]:
            pth = _check_HDF_file(file)
            h5data = h5py.File(pth, 'r')

            h5master['entry1/data/hmm'][0] += \
                h5data['entry1/data/hmm'][0]
            h5master['entry1/monitor/bm1_counts'][0] += \
                h5data['entry1/monitor/bm1_counts'][0]
            h5master['entry1/instrument/detector/total_counts'][0] += \
                h5data['entry1/instrument/detector/total_counts'][0]
            h5master['entry1/instrument/detector/time'][0] += \
                h5data['entry1/instrument/detector/time'][0]

            h5master.flush()


def _check_HDF_file(h5data):
    if type(h5data) == h5py.File:
        return h5data.filename
    else:
        with h5py.File(h5data, 'r') as h5data:
            if type(h5data) == h5py.File:
                return h5data.filename

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some Platypus NeXUS'
                                                 'files to produce their TOF '
                                                 'spectra.')

    parser.add_argument('file_list', metavar='N', type=int, nargs='+',
                        help='integer file numbers')

    parser.add_argument('-b', '--bdir', type=str,
                        help='define the location to find the nexus files')

    parser.add_argument('-d', '--direct', action='store_true', default=False,
                        help='is the file a direct beam?')

    parser.add_argument('-r', '--rebin', type=float,
                        help='rebin percentage for the wavelength -1<rebin<10',
                        default=1)

    parser.add_argument('-ll', '--lolambda', type=float,
                        help='lo wavelength cutoff for the rebinning',
                        default=2.5)

    parser.add_argument('-hl', '--hilambda', type=float,
                        help='lo wavelength cutoff for the rebinning',
                        default=19.)

    parser.add_argument('-i', '--integrate', type=int,
                        help='-1 to integrate all spectra, otherwise enter the'
                             ' spectrum number.', default=-1)
    args = parser.parse_args()

    for file in args.file_list:
        fname = 'PLP%07d.nx.hdf' % file
        path = os.path.join(args.bdir, fname)
        try:
            a = PlatypusNexus(path)
            a.process(lo_wavelength=args.lolambda,
                      hi_wavelength=args.hilambda,
                      direct=args.direct,
                      rebin_percent=args.rebin,
                      integrate=args.integrate)

            fname = 'PLP%07d.spectrum' % file
            out_fname = os.path.join(args.bdir, fname)

            integrate = args.integrate
            if args.integrate < 0:
                integrate = 0

            a.write_spectrum_dat(out_fname, scanpoint=integrate)

        except IOError:
            print("Couldn't find file: %d.  Use --basedir option" % file)
