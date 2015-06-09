from __future__ import division
import numpy as np
import h5py
import peak_utils as ut
import refnx.util.general as general
import refnx.util.ErrorProp as EP
import platypusspectrum
import event
import Qtransforms as qtrans
from scipy.optimize import curve_fit
from scipy.stats import t
import rebin
import os
import os.path
import argparse
import re


Y_PIXEL_SPACING = 1.177  # in mm

disc_openings = (60., 10., 25., 60.)
O_C1d, O_C2d, O_C3d, O_C4d = disc_openings
O_C1, O_C2, O_C3, O_C4 = np.radians(disc_openings)

DISCRADIUS = 350.
EXTENT_MULT = 2
PIXEL_OFFSET = 1

class Catalogue(object):
    def __init__(self, h5data):
        d = {}
        path = os.path.realpath(h5data.filename)
        d['path'] = os.path.dirname(path)
        d['filename'] = h5data.filename
        d['end_time'] = h5data['entry1/end_time'][0]
        d['ss1vg'] = h5data['entry1/instrument/slits/first/vertical/gap'][:]
        d['ss2vg'] = h5data['entry1/instrument/slits/second/vertical/gap'][:]
        d['ss3vg'] = h5data['entry1/instrument/slits/third/vertical/gap'][:]
        d['ss4vg'] = h5data['entry1/instrument/slits/fourth/vertical/gap'][:]
        d['ss1hg'] = h5data['entry1/instrument/slits/first/horizontal/gap'][:]
        d['ss2hg'] = h5data['entry1/instrument/slits/second/horizontal/gap'][:]
        d['ss3hg'] = h5data['entry1/instrument/slits/third/horizontal/gap'][:]
        d['ss4hg'] = h5data['entry1/instrument/slits/fourth/horizontal/gap'][:]

        d['detector'] = h5data['entry1/data/hmm'][:]
        d['sth'] = h5data['entry1/sample/sth'][:]
        d['bm1_counts'] = h5data['entry1/monitor/bm1_counts'][:]
        d['total_counts'] = h5data['entry1/instrument/detector/total_counts'][:]
        d['time'] = h5data['entry1/instrument/detector/time'][:]
        d['mode'] = h5data['entry1/instrument/parameters/mode'][0]

        try:
            event_directory_name = h5data[
                'entry1/instrument/detector/daq_dirname'][0]
            d['daq_dirname'] = event_directory_name
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
        d['dy'] = h5data[
            'entry1/instrument/detector/longitudinal_translation'][:]
        d['dz'] = h5data[
            'entry1/instrument/detector/vertical_translation'][:]
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

        if ('entry1/instrument/parameters/slave' in h5data
            and 'entry1/instrument/parameters/master' in h5data):
            master = h5data['entry1/instrument/parameters/master']
            slave = h5data['entry1/instrument/parameters/slave']
        else:
            master = 1
            if abs(chopper2_speed) > 10:
                slave = 2
            elif abs(chopper3_speed) > 10:
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

        return master[:], slave[:], speeds[0] / 60., phases[slave[0] - 1]


def datafile_number(fname):
    regex = re.compile("PLP([0-9]{7}).nx.hdf")
    _fname = os.path.basename(fname)
    r = regex.search(_fname)

    if r:
        return int(r.groups()[0])

    return None


class PlatypusNexus(object):
    """
    Processes Platypus NeXus files to produce an intensity vs wavelength
    spectrum
    """

    def __init__(self, h5data):
        """
        Initialises the PlatypusNexus object.

        Parameters
        ----------
        h5data : HDF5 NeXus file or str
            An HDF5 NeXus file for Platypus, or a `str` containing the path
            to one.
        """
        if type(h5data) == h5py.File:
            self.cat = Catalogue(h5data)
        else:
            with h5py.File(h5data, 'r') as h5data:
                self.cat = Catalogue(h5data)

    def process(self, h5norm=None, lo_wavelength=2.8, hi_wavelength=19.,
                background=True, is_direct=False, omega=0, two_theta=0,
                rebin_percent=1., wavelength_bins=None, normalise=True, integrate=0,
                eventmode=None, peak_pos=None, background_mask=None):
        """
        Processes the ProcessNexus object to produce a time of flight spectrum.
        This method returns an instance of PlatypusSpectrum.

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
        is_direct : bool
            Is it a direct beam you measured? This is so a gravity correction
            can be applied.
        omega : float
            Expected angle of incidence of beam
        two_theta : float
            Expected two theta value of specular beam
        rebin_percent : float
            Specifies the rebinning percentage for the spectrum.  If `rebinning
            is None`, then no rebinning is done.
        wavelength_bins : array_like
            The wavelength bins for rebinning.  If `wavelength_bins is not
             None` then the `rebin_percent` parameter is ignored.
        normalise : bool
            Normalise by the monitor counts.
        integrate : int
            integrate == 0
                the spectrum is integrated over all the scanpoints.
            integrate != 0
                the individual spectra are calculated individually.
                If `eventmode is not None` then integrate specifies which
                scanpoint to examine.
        eventmode : None or array_like
            If eventmode is `None` then the integrated detector image is used.
            If eventmode is an array then the array specifies the integration
            times (in seconds) for the detector image, e.g. [0, 20, 30] would
            result in two spectra. The first would contain data for 0 s to 20s,
            the second would contain data for 20 s to 30 s.  This option can
            only be used when `integrate != 0`.
        peak_pos : None or (float, float)
            Specifies the peak position and peak standard deviation to use.
        background_mask : array_like
            An array of bool that specifies which y-pixels to use for background
            subtraction.  Should be the same length as the number of y pixels in
            the detector image.  Otherwise an automatic mask is applied (if
            background is True).
        """
        cat = self.cat

        scanpoint = 0

        # beam monitor counts for normalising data
        bm1_counts = cat.bm1_counts.astype('float64')

        # TOF bins
        TOF = cat.t_bins.astype('float64')

        # We want event streaming.
        if eventmode is not None and integrate > 0:
            scanpoint = integrate
            output = self.process_event_stream(scanpoint=scanpoint,
                                               frame_bins=eventmode)
            frame_bins, detector, bm1_counts = output

        else:
            detector = cat.detector
            scanpoint = 0

            # integrate over all spectra
            if integrate == 0:
                detector = np.sum(detector, 0)[np.newaxis, ]
                bm1_counts[:] = np.sum(bm1_counts)

        num_spectra = np.size(detector, 0)

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
            # broadcast to (1,1,y)
            detector, detector_sd = EP.EPdiv(detector, detector_sd,
                                             detector_norm, detector_norm_sd)

        # shape of these is (num_spectra, TOFbins)
        M_spec_tof_hist = np.zeros((num_spectra, np.size(TOF, 0)),
                                   dtype='float64')
        M_lambda_hist = np.zeros((num_spectra, np.size(TOF, 0)),
                                 dtype='float64')
        M_spec_tof_hist[:] = TOF

        # chopper to detector distances
        # note that if eventmode is specified the num_spectra is NOT
        # equal to the number of entries in e.g. /longitudinal_translation
        # this means you have to copy values in from the correct scanpoint
        flight_distance = np.zeros(num_spectra, dtype='float64')
        d_cx = np.zeros(num_spectra, dtype='float64')
        detpositions = np.zeros(num_spectra, dtype='float64')

        # The angular divergence of the instrument
        domega = np.zeros(num_spectra, dtype='float64')

        phase_angle = np.zeros(num_spectra, dtype='float64')

        # process each of the spectra taken in the detector image
        originalscanpoint = scanpoint
        for idx in range(num_spectra):
            freq = cat.frequency[scanpoint]

            # calculate the angular divergence
            domega[idx] = general.div(cat.ss2vg[scanpoint],
                                      cat.ss3vg[scanpoint],
                                      (cat.slit3_distance[0]
                                       - cat.slit2_distance[0]))[0]

            # work out the total flight length
            output = self.chod(omega, two_theta, scanpoint=scanpoint)
            flight_distance[idx], d_cx[idx] = output

            # calculate phase openings
            output = self.phase_angle(scanpoint)
            phase_angle[scanpoint], master_opening = output

            # toffset - the time difference between the magnet pickup on the
            # choppers (TTL pulse), which is situated in the middle of the
            # chopper window, and the trailing edge of chopper 1, which is
            # supposed to be time0.  However, if there is a phase opening this
            # time offset has to be relocated slightly, as time0 is not at the
            # trailing edge.
            poff = cat.chopper1_phase_offset[0]
            poffset = 1.e6 * poff / (2. * 360. * freq)
            toffset = (poffset
                       + 1.e6 * master_opening / 2 / (2 * np.pi) / freq
                       - 1.e6 * phase_angle / (360 * 2 * freq))
            M_spec_tof_hist[idx] -= toffset

            detpositions[idx] = cat.dy[scanpoint]

            if eventmode is not None and integrate > 0:
                M_spec_tof_hist[:] = TOF - toffset
                flight_distance[:] = flight_distance[0]
                detpositions[:] = detpositions[0]
                break
            else:
                scanpoint += 1

        scanpoint = originalscanpoint

        # convert TOF to lambda
        # M_spec_tof_hist (n, t) and chod is (n,)
        M_lambda_hist = qtrans.tof_to_lambda(M_spec_tof_hist,
                                             flight_distance[:, np.newaxis])
        M_lambda = 0.5 * (M_lambda_hist[:, 1:] + M_lambda_hist[:, :-1])
        TOF -= toffset

        # get the specular ridge on the averaged detector image
        if peak_pos is not None:
            beam_centre, beam_sd = peak_pos
        else:
            beam_centre, beam_sd = find_specular_ridge(detector, detector_sd)

        # gravity correction if direct beam
        if is_direct:
            output = correct_for_gravity(detector,
                                         detector_sd,
                                         M_lambda,
                                         0,
                                         lo_wavelength,
                                         hi_wavelength)
            detector, detector_sd, M_gravcorrcoefs = output

            beam_centre, beam_sd = find_specular_ridge(detector, detector_sd)

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
            rebinning = M_lambda_hist[0, :]
            rebinning = rebinning[np.searchsorted(rebinning, lo_wavelength):
                                  np.searchsorted(rebinning, hi_wavelength)]

        '''
        now do the rebinning for all the N detector images
        rebin.rebinND could do all of these at once.  However, M_lambda_hist
        could vary across the range of spectra.  If it was the same I could
        eliminate the loop.
        '''
        output = []
        output_sd = []
        for idx in range(num_spectra):
            plane, plane_sd = rebin.rebin_along_axis(detector[idx],
                                                     M_lambda_hist[idx],
                                                     rebinning,
                                                     y1_sd=detector_sd[idx])
            output.append(plane)
            output_sd.append(plane_sd)

        detector = np.vstack(output)
        detector_sd = np.vstack(output_sd)

        if len(detector.shape) == 2:
            detector = detector[np.newaxis, ]
            detector_sd = detector_sd[np.newaxis, ]

        #(1, T)
        M_lambda_hist = np.atleast_2d(rebinning)

        '''
        Divide the detector intensities by the width of the wavelength bin.
        This is so the intensities between different rebinning strategies can
        be compared.
        '''
        div = 1 / np.ediff1d(M_lambda_hist[0])[:, np.newaxis]
        detector, detector_sd = EP.EPmulk(detector,
                                          detector_sd,
                                          div)

        # convert the wavelength base to a timebase
        M_spec_tof_hist = qtrans.lambda_to_tof(M_lambda_hist,
                                               flight_distance[:, np.newaxis])

        M_lambda = 0.5 * (M_lambda_hist[:, 1:] + M_lambda_hist[:, :-1])

        M_spec_tof = qtrans.lambda_to_tof(M_lambda,
                                          flight_distance[:, np.newaxis])

        '''
        Now work out where the beam hits the detector
        this is used to work out the correct angle of incidence.
        '''
        M_beampos = np.ones_like(M_lambda) * beam_centre

        '''
        The spectral ridge for the direct beam has a gravity correction
        involved with it. The correction coefficients for the beamposition
        are contained in M_gravcorrcoefs.
        '''
        if is_direct:
            M_beampos[:] = self._beampos(M_lambda,
                                         M_gravcorrcoefs,
                                         detpositions)
        M_beampos[:] *= Y_PIXEL_SPACING

        # we want to integrate over the following pixel region
        lopx = np.floor(beam_centre - beam_sd * EXTENT_MULT)
        hipx = np.ceil(beam_centre + beam_sd * EXTENT_MULT)

        # background subtraction
        if background:
            if background_mask is not None:
                pass
            else:
                y1 = round(lopx - PIXEL_OFFSET)
                y0 = round(y1 - (EXTENT_MULT * beam_sd))

                y2 = round(hipx + PIXEL_OFFSET)
                y3 = round(y2 + (EXTENT_MULT * beam_sd))

                background_mask = np.zeros(detector.shape[2], dtype='bool')
                background_mask[y0: y1] = True
                background_mask[y2 + 1: y3 + 1] = True

            detector, detector_sd = background_subtract(detector,
                                                        detector_sd,
                                                        background_mask)

        '''
        top and tail the specular beam with the known beam centres.
        All this does is produce a specular intensity with shape (N, T),
        i.e. integrate over specular beam
        '''
        M_spec = np.sum(detector[:, :, lopx: hipx + 1], axis=2)
        M_spec_sd = np.sqrt(np.sum(detector_sd[:, :, lopx: hipx + 1]**2,
                                   axis=2))
        # assert np.isfinite(M_spec).all()
        # assert np.isfinite(M_specSD).all()
        # assert np.isfinite(detector).all()
        # assert np.isfinite(detectorSD).all()

        # normalise by beam monitor 1.
        if normalise:
            # have to make to the same shape as M_spec
            M_spec, M_spec_sd = EP.EPdiv(M_spec,
                                         M_spec_sd,
                                         bm1_counts[:, np.newaxis],
                                         bm1_counts_sd[:, np.newaxis])

            # have to make to the same shape as detector
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
        tau_da = M_spec_tof_hist[:, 1:] - M_spec_tof_hist[:, :-1]

        M_lambda_sd = general.resolution_double_chopper(M_lambda,
                                     z0=d_cx[:, np.newaxis] / 1000.,
                                     freq=cat.frequency[:, np.newaxis],
                                     L=flight_distance[:, np.newaxis] / 1000.,
                                     H=cat.ss2vg[originalscanpoint],
                                     xsi=phase_angle[:, np.newaxis],
                                     tau_da=tau_da)

        M_lambda_sd *= M_lambda

        # put the detector positions and mode into the dictionary as well.
        detectorZ = np.atleast_2d(cat.dz)
        detectorY = np.atleast_2d(cat.dy)
        mode = np.atleast_2d(cat.mode)

        d = dict()
        d['path'] = cat.path
        d['datafilename'] = cat.filename
        d['datafile_number'] = cat.datafile_number

        if h5norm is not None:
            d['normfilename'] = h5norm.filename
        d['M_topandtail'] = detector
        d['M_topandtail_sd'] = detector_sd
        d['num_spectra'] = num_spectra
        d['bm1_counts'] = bm1_counts
        d['M_spec'] = M_spec
        d['M_spec_sd'] = M_spec_sd
        d['M_beampos'] = M_beampos
        d['M_lambda'] = M_lambda
        d['M_lambda_sd'] = M_lambda_sd
        d['M_lambda_hist'] = M_lambda_hist
        d['M_spec_tof'] = M_spec_tof
        d['mode'] = mode
        d['detectorZ'] = detectorZ
        d['detectorY'] = detectorY
        d['domega'] = domega
        d['lopx'] = lopx
        d['hipx'] = hipx

        return platypusspectrum.PlatypusSpectrum(**d)


    def _beampos(self, lamda, M_gravcorrcoefs, dy):
        """
        Workout beam position for a droopy beam

        Parameters
        ----------
        lamda : array_like
            wavelength (Angstrom)
        M_gravcorrcoefs : array_like
            Gravity correction coefficients
        dy : array
            Detector horizontal translation

        Returns
        -------
        beam_pos : array
            beam positions

        The following correction assumes that the directbeam neutrons are
        falling from a point position W_gravcorrcoefs[0] before the
        detector. At the sample stage
        (W_gravcorrcoefs[0] - detectorpos[0]) they have a certain vertical
        velocity, assuming that the neutrons had an initial vertical
        velocity of 0. Although the motion past the sample stage will be
        parabolic, assume that the neutrons travel in a straight line
        after that (i.e. the tangent of the parabolic motion at the
        sample stage). This should give an idea of the direction of the
        true incident beam, as experienced by the sample.
        Factor of 2 is out the front to give an estimation of the
        increase in 2theta of the reflected beam.
        """
        M_beampos = np.zeros_like(lamda)
        M_beampos[:] = M_gravcorrcoefs[:, 1][:, np.newaxis]
        M_beampos[:] -= 2. * (1000. / Y_PIXEL_SPACING * 9.81
                              * (M_gravcorrcoefs[:, 0][:, np.newaxis]
                                 - dy[:, np.newaxis]) / 1000.
                              * (dy[:, np.newaxis] / 1000.)
                              * lamda ** 2
                              / ((qtrans.kPlanck_over_MN * 1.e10) ** 2))
        return M_beampos

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
        master = cat.master[scanpoint]
        slave = cat.slave[scanpoint]
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

    def chod(self, omega=0., two_theta=0., scanpoint=0):
        """
        Calculates the flight length of the neutrons in the Platypus instrument.

        Parameters
        ----------
        omega : float, optional
            Rough angle of incidence
        two_theta : float, optional
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
        master = cat.master[scanpoint]
        slave = cat.slave[scanpoint]

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
            chod += cat.dy[scanpoint] / np.cos(np.radians(two_theta))

        elif mode == 'SB':
            # assumes guide1_distance is in the MIDDLE OF THE MIRROR
            chod += cat.guide1_distance[0]
            chod += ((cat.sample_distance[0] - cat.guide1_distance[0])
                     / np.cos(np.radians(omega)))
            if two_theta > omega:
                chod += (cat.dy[scanpoint] /
                         np.cos(np.radians(two_theta - omega)))
            else:
                chod += (cat.dy[scanpoint]
                         / np.cos(np.radians(omega - two_theta)))

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
            if two_theta > omega:
                chod += (cat.dy[scanpoint]
                         / np.cos(np.radians(two_theta - 4.8)))
            else:
                chod += (cat.dy[scanpoint]
                         / np.cos(np.radians(4.8 - two_theta)))

        return chod, d_cx

    def process_event_stream(self, t_bins=None, x_bins=None, y_bins=None,
                             frame_bins=None, scanpoint=0):
        """
        Processes the event mode dataset for the nexus file. Assumes that
        there is a event mode directory in the same directory as the NeXUS
        file, as specified by in 'entry1/instrument/detector/daq_dirname'

        Parameters
        ----------
        frame_bins : array_like, optional
            specifies the frame bins required in the image. If
            framebins = [5, 10, 120] you will get 2 images.  The first starts
            at 5s and finishes at 10s. The second starts at 10s and finishes
            at 120s.
        t_bins : array_like, optional
            specifies the time bins required in the image
        x_bins : array_like, optional
            specifies the x bins required in the image
        y_bins : array_like, optional
            specifies the y bins required in the image
        scanpoint : int, optional
            Scanpoint you are interested in

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
        if not frame_bins:
            frame_bins = [0, cat.time[scanpoint]]

        total_acquisition_time = cat.time[scanpoint]
        frequency = cat.frequency[scanpoint]

        bm1_counts_for_scanpoint = cat.bm1_counts[scanpoint]

        event_directory_name = cat['daq_dirname'][0]

        stream_filename = os.path.join(cat.path,
                                       event_directory_name,
                                       'DATASET_%d' % scanpoint,
                                       'EOS.bin')

        with open(stream_filename, 'r') as f:
            events, end_of_last_event = event.events(f)

        output = event.process_event_stream(events,
                                            frame_bins * frequency,
                                            t_bins,
                                            y_bins,
                                            x_bins)

        detector, new_frame_bins = output

        new_frame_bins /= frequency

        bm1_counts = new_frame_bins[1:] - new_frame_bins[:-1]
        bm1_counts *= (bm1_counts_for_scanpoint / total_acquisition_time)

        return new_frame_bins, detector, bm1_counts


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
    (N, T) slice

    Parameters
    ----------
    detector : np.ndarray
        detector array with shape (N, T, Y).
    detector_sd : np.ndarray
        standard deviations for detector array
    background_mask : array_like
        array of bool that specifies which Y pixels to use for background
        subtraction.

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
                                                         background_mask)
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

    y_vals = profile[x_vals]
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
    popt, pcov = curve_fit(f, x_vals, y_vals, sigma=y_sdvals,
                           p0=np.array([ahat, bhat]))

    def CI(xx, pcovmat):
        return (pcovmat[0, 0] + pcovmat[1, 0] * xx
                + pcovmat[0, 1] * xx + pcovmat[1, 1] * (xx ** 2))

    bkgd = f(np.arange(np.size(profile, 0)), popt[0], popt[1])
    bkgd_sd = np.empty_like(bkgd)

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
    Find the specular ridge in a detector(n, t, y) plot. Assumes that the
    specular ridge _does not_ change position.

    Parameters
    ----------
    detector : array_like
        detector array
    detector_sd : array_like
        standard deviations of detector array

    Returns
    -------
    centre, SD:
        peak centre and standard deviation of peak width
    """

    search_increment = 50

    # if [N,T,Y] sum over all N planes, left with [T, Y]
    if len(detector.shape) > 2:
        det_ty = np.sum(detector, axis=0)
        det_sd_ty = np.sqrt(np.sum(detector_sd**2, axis=0))
    else:
        det_ty, det_sd_ty = detector, detector_sd

    starting_offset = abs(starting_offset)

    num_increments = (np.size(det_ty, 0) - starting_offset) // search_increment

    last_centre = -1.
    last_sd = -1.

    for i in range(num_increments):
        how_many = -starting_offset - search_increment * i

        det_subset = det_ty[-1: how_many: -1]
        det_sd_subset = det_sd_ty[-1: how_many: -1]

        # Uncertainties code takes a while to run
        # total_y = np.sum(det_subset, axis=0)
        y_cross = np.sum(det_subset, axis=0)
        y_cross_sd = np.sqrt(np.sum(det_sd_subset ** 2., axis=0))

        # find the centroid and gauss peak in the last sections of the TOF
        # plot
        centroid, gauss_peak = ut.peak_finder(y_cross, sigma=y_cross_sd)

        if (abs((gauss_peak[0] - last_centre) / last_centre) < tolerance
            and abs((gauss_peak[1] - last_sd) / last_sd) < tolerance):
            last_centre = gauss_peak[0]
            last_sd = gauss_peak[1]
            break

        last_centre = gauss_peak[0]
        last_sd = gauss_peak[1]

    return last_centre, last_sd


def correct_for_gravity(detector, detector_sd, lamda, trajectory,
                        lo_wavelength, hi_wavelength):
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
    trajectory : float
        Initial trajectory of neutrons
    lo_wavelength : float
        Low wavelength cut off
    hi_wavelength : float
        High wavelength cutoff

    Returns
    -------
    corrected_data, corrected_data_sd, M_gravcorrcoefs :
                    np.ndarray, np.ndarray, np.ndarray
        Corrected image. This is a theoretical prediction where the spectral
        ridge is for each wavelength.  This will be used to calculate the
        actual angle of incidence in the reduction process.

    """
    num_lambda = np.size(lamda, axis=1)

    x_init = np.arange((np.size(detector, axis=2) + 1) * 1.) - 0.5

    def f(x, travel_distance, tru_centre):
        return deflection(x, travel_distance, 0) / Y_PIXEL_SPACING + tru_centre

    M_gravcorrcoefs = np.zeros((np.size(detector, 0), 2), dtype='float64')

    corrected_data = np.zeros_like(detector)
    corrected_data_sd = np.zeros_like(detector)

    for spec in range(np.size(detector, 0)):
        # centres(t,)
        # TODO, don't use centroids, use Gaussian peak
        centroids = np.apply_along_axis(ut.centroid, 1, detector[spec])
        lopx = np.searchsorted(lamda[spec], lo_wavelength)
        hipx = np.searchsorted(lamda[spec], hi_wavelength)

        p0 = np.array([3000., np.mean(centroids)])
        M_gravcorrcoefs[spec], pcov = curve_fit(f, lamda[spec, lopx: hipx],
                                                centroids[:, 0][lopx: hipx],
                                                p0)
        total_deflection = deflection(lamda[spec],
                                      M_gravcorrcoefs[spec][0],
                                      0) / Y_PIXEL_SPACING

        x_rebin = x_init.T + total_deflection[:, np.newaxis]
        for wavelength in range(np.size(detector, axis=1)):
            output = rebin.rebin(x_init,
                                 detector[spec, wavelength],
                                 x_rebin[wavelength],
                                 y1_sd=detector_sd[spec, wavelength])

            corrected_data[spec, wavelength] = output[0]
            corrected_data_sd[spec, wavelength] = output[1]

    return corrected_data, M_gravcorrcoefs


def deflection(lamda, flight_length, trajectory):
    """
    The vertical deflection in mm of a ballistic neutron after travelling a
    certain distance.

    Parameters
    ----------
    lamda : float
        wavelength of neutron in Angstrom
    flight_length : float
        Flight length in mm,
    trajectory : float
        initial trajectory of neutron in degrees above the horizontal.

    The deflection correction is the distance from where you expect the
    neutron to hit the detector (detector_distance*tan(trajectory)) to
    where is actually hits the detector, i.e. the vertical deflection of
    the neutron due to gravity.
    """
    traj = np.radians(trajectory)

    initial_velocity = general.wavelength_velocity(lamda)

    # x = v_0 . t . cos(trajectory)
    # y = v_0 . t . sin(trajectory) - 0.5gt^2

    flight_time = flight_length / 1000. / (initial_velocity * np.cos(traj))

    y_t = (initial_velocity * flight_time * np.sin(traj)
           - 0.5 * 9.81 * flight_time ** 2)

    return y_t * 1000.


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
    num_steps = np.floor(np.log10(hil / lowl) / np.log10(frac)) + 1
    rebinning = np.logspace(np.log10(lowl), np.log10(hil), num=num_steps)
    return rebinning


# def catalogue_all(basedir=None, fname=None):
#     if not basedir:
#         files_to_catalogue = [filename for filename in os.listdir(os.getcwd()) if is_platypus_file(filename)]
#     else:
#         files_to_catalogue = []
#         for root, dirs, files in os.walk(basedir):
#             files_to_catalogue.append(
#                 [os.path.join(root, filename) for filename in files if is_platypus_file(filename)])
#
#     files_to_catalogue = [item for sublist in files_to_catalogue for item in sublist]
#     filenumbers = [is_platypus_file(filename) for filename in files_to_catalogue]
#
#     Tppn = ProcessPlatypusNexus()
#
#     listdata = []
#
#     for filename in files_to_catalogue:
#         try:
#             with h5.File(filename, 'r') as h5data:
#                 listdata.append((is_platypus_file(filename), Tppn.catalogue(h5data)))
#                 h5data.close()
#         except:
#             pass
#
#     uniquelist = []
#     uniquefnums = []
#     for item in listdata:
#         if not item[0] in uniquefnums:
#             uniquelist.append(item)
#             uniquefnums.append(item[0])
#
#     uniquelist.sort()
#     if fname:
#         template = """$datafilenumber\t$end_time\t$ss1vg\t$ss2vg\t$ss3vg\t$ss4vg\t$total_counts\t$bm1_counts\t$time\t$mode\t$daq_dirname\n"""
#         with open(fname, 'w') as f:
#             f.write(template)
#             s = string.Template(template)
#
#             for item in uniquelist:
#                 f.write(s.safe_substitute(item[1]))
#
#             f.truncate()
#
#     return uniquelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some Platypus NeXUS files to produce their TOF spectra.')
    parser.add_argument('file_list', metavar='N', type=int, nargs='+',
                        help='integer file numbers')
    parser.add_argument('-b', '--basedir', type=str, help='define the location to find the nexus files')
    parser.add_argument('-r', '--rebinpercent', type=float, help='rebin percentage for the wavelength -1<rebin<10', default=1)
    parser.add_argument('-l', '--lolambda', type=float, help='lo wavelength cutoff for the rebinning', default=2.5)
    parser.add_argument('--hilambda', type=float, help='lo wavelength cutoff for the rebinning', default=19.)
    parser.add_argument('--typeofintegration', type=float,
                        help='0 to integrate all spectra, 1 to output individual spectra', default=0)
    args = parser.parse_args()
    print args

    for file in args.file_list:
        fname = 'PLP%07d.nx.hdf' % file
        path = os.path.join(args.basedir, fname)
        try:
            a = PlatypusNexus(path)
            a.process()

            # M_lambda, M_lambdaSD, M_spec, M_specSD = a.process(lolambda=args.lolambda,
            #                                                    hilambda=args.hilambda,
            #                                                    rebinpercent=args.rebin,
            #                                                    typeofintegration=args.typeofintegration)
            #
            # for index in xrange(a.numspectra):
            #     filename = 'PLP{:07d}_{:d}.spectrum'.format(a.datafilenumber, index)
            #     f = open(filename, 'w')
            #     a.writespectrum(f, scanpoint=index)
            #     f.close()

        except IOError:
            print "Couldn't find file: %d.  Use --basedir option" % file
        

