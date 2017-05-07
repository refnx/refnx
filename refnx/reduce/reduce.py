from __future__ import division
import string
from copy import deepcopy
import os.path
from time import gmtime, strftime

import numpy as np
from refnx.reduce.platypusnexus import (PlatypusNexus, number_datafile,
                                        Y_PIXEL_SPACING, basename_datafile)
from refnx.util import ErrorProp as EP
import refnx.util.general as general
from .parabolic_motion import (parabola_line_intersection_point,
                               find_trajectory)
from refnx.dataset import ReflectDataset


_template_ref_xml = """<?xml version="1.0"?>
<REFroot xmlns="">
<REFentry time="$time">
<Title>$title</Title>
<User>$user</User>
<REFsample>
<ID>$sample</ID>
</REFsample>
<REFdata axes="Qz:Qy" rank="2" type="POINT" \
spin="UNPOLARISED" dim="$_numpointsz:$_numpointsy">
<Run filename="$_rnumber" preset="" size="">
</Run>
<R uncertainty="dR">$_r</R>
<Qz uncertainty="dQz" units="1/A">$_qz</Qz>
<dR type="SD">$_dr</dR>
<Qy type="_FWHM" units="1/A">$_qy</Qy>
</REFdata>
</REFentry>
</REFroot>"""


class PlatypusReduce(object):
    """
    Reduces Platypus reflectometer data to give the specular reflectivity.
    Offspecular data maps are also produced.

    Parameters
    ----------
    direct : string, hdf5 file-handle or PlatypusNexus object
        A string containing the path to the direct beam hdf5 file,
        the hdf5 file itself, or a PlatypusNexus object.
    reflect : string, hdf5 file-handle or PlatypusNexus object, optional
        A string containing the path to the specularly reflected hdf5 file,
        the hdf5 file itself, or a PlatypusNexus object.
    data_folder : str, optional
        Where is the raw data stored?
    scale : float, optional
        Divide all specular reflectivity values by this number.
    save : bool, optional
        If `True` then the reduced dataset is saved to the current
        directory, with a name os.path.basename(reflect)
    kwds : dict, optional
        Options passed directly to `refnx.reduce.platypusnexus.process`,
        for processing of individual spectra. Look at that method docstring
        for specification of options.

    Notes
    -----
    If `reflect` was specified during construction a reduction will be
    done. See ``reduce`` for attributes available from this object on
    completion of reduction.

    Returns
    -------
    None
    """

    def __init__(self, direct, reflect=None, data_folder=None, scale=1.,
                 save=True, **kwds):
        self.data_folder = os.path.curdir
        if data_folder is not None:
            self.data_folder = data_folder

        if isinstance(direct, PlatypusNexus):
            self.direct_beam = direct
        elif type(direct) is str:
            direct = os.path.join(self.data_folder, direct)
            self.direct_beam = PlatypusNexus(direct)
        else:
            self.direct_beam = PlatypusNexus(direct)

        if reflect is not None:
            self.reduce(reflect, save=save, scale=scale, **kwds)

    def __call__(self, reflect, scale=1, save=True, **kwds):
        return self.reduce(reflect, scale=scale, save=save, **kwds)

    def reduce(self, reflect, scale=1., save=True, **kwds):
        """
        Reduction of a single dataset.

        The reduction uses the direct beam specified during construction of
        this object. This method reduces all the spectra present in the
        reflected beam file (see platypusnexus.PlatypusNexus.process for
        eventmode specification and other related options), but aggregates
        all data in the direct beam spectrum.

        Parameters
        ----------
        reflect : string, hdf5 file-handle or PlatypusNexus object
            A string containing the path to the specularly reflected hdf5 file,
            the hdf5 file itself, or a PlatypusNexus object.
        scale : float, optional
            Divide all the reflectivity values by this number.
        save : bool, optional
            If `True` then the reduced dataset is saved to the current
            directory, with a name os.path.basename(reflect)
        kwds : dict, optional
            Options passed directly to `refnx.reduce.platypusnexus.process`,
            for processing of individual spectra. Look at that method docstring
            for specification of options.

        Returns
        -------
        reduction : dict
            Contains the following entries:

            - 'xdata' : np.ndarray
                Q values, shape (N, T).
            - 'xdata_sd' : np.ndarray
                Uncertainty in Q values (FWHM), shape (N, T).
            - 'ydata' : np.ndarray
                Specular Reflectivity, shape (N, T)
            - 'ydata_sd' : np.ndarray
                Uncertainty in specular reflectivity (SD), shape (N, T)
            - 'omega' : np.ndarray
                Angle of incidence, shape (N, T)
            - 'm_ref' : np.ndarray
                Offspecular reflectivity map, shape (N, T, Y)
            - 'm_ref_sd' : np.ndarray
                uncertainty in offspecular reflectivity, shape (N, T, Y)
            - 'm_qz' : np.ndarray
                Qz for offspecular map, shape (N, T, Y)
            - 'm_qy' : np.ndarray
                Qy for offspecular map, shape (N, T, Y)
            - 'n_spectra' : int
                number of reflectivity spectra
            - 'datafile_number' : int
                run number for the reflected beam
            - 'fname' : list
                the saved filenames

        N corresponds to the number of spectra
        T corresponds to the number of Q (wavelength) bins
        Y corresponds to the number of y pixels on the detector.

        Notes
        -----
        All the values returned from this method are also contained as instance
        attributes for this object.

        Examples
        --------

        >>> from refnx.reduce import PlatypusReduce
        >>> # set up with a direct beam
        >>> reducer = PlatypusReduce('PLP0000711.nx.hdf')
        >>> reduction = reducer.reduce('PLP0000708.nx.hdf', rebin_percent=3.)
        """
        reflect_keywords = kwds.copy()
        direct_keywords = kwds.copy()

        # get the direct beam spectrum
        direct_keywords['direct'] = True
        direct_keywords['integrate'] = -1

        if 'eventmode' in direct_keywords:
            direct_keywords.pop('eventmode')

        self.direct_beam.process(**direct_keywords)

        # get the reflected beam spectrum
        reflect_keywords['direct'] = False
        if isinstance(reflect, PlatypusNexus):
            self.reflected_beam = reflect
        elif type(reflect) is str:
            reflect = os.path.join(self.data_folder, reflect)
            self.reflected_beam = PlatypusNexus(reflect)
        else:
            self.reflected_beam = PlatypusNexus(reflect)

        # Got to use the same wavelength bins as the direct spectrum.
        # done this way around to save processing direct beam over and over
        reflect_keywords['wavelength_bins'] = self.direct_beam.m_lambda_hist[0]

        self.reflected_beam.process(**reflect_keywords)

        self.save = save
        reduction = self._reduce_single_angle(scale)
        return reduction

    def data(self, scanpoint=0):
        """
        The specular reflectivity

        Parameters
        ----------
        scanpoint: int
            Find a particular specular reflectivity image. scanpoints upto
            `self.n_spectra - 1` can be specified.

        Returns
        -------
        (Q, R, dR, dQ): np.ndarray tuple
            dR is standard deviation, dQ is FWHM
        """
        return (self.xdata[scanpoint],
                self.ydata[scanpoint],
                self.ydata_sd[scanpoint],
                self.xdata_sd[scanpoint])

    def data2d(self, scanpoint=0):
        """
        The offspecular data

        Parameters
        ----------
        scanpoint: int
            Find a particular offspecular image. scanpoints upto
            self.n_spectra - 1 can be specified.

        Returns
        -------
        (Qz, Qy, R, dR): np.ndarrays
        """

        return (self.m_qz[scanpoint],
                self.m_qy[scanpoint],
                self.m_ref[scanpoint],
                self.m_ref_sd[scanpoint])

    def scale(self, scale):
        """
        Divides the reflectivity values by this scale factor

        Parameters
        ----------
        scale: float
            Divides the reflectivity values by a constant amount
        """
        self.m_ref /= scale
        self.m_ref_sd /= scale
        self.ydata /= scale
        self.ydata_sd /= scale

    def write_offspecular(self, f, scanpoint=0):
        d = dict()
        d['time'] = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        d['_rnumber'] = self.reflected_beam.datafile_number
        d['_numpointsz'] = np.size(self.m_ref, 1)
        d['_numpointsy'] = np.size(self.m_ref, 2)

        s = string.Template(_template_ref_xml)

        # filename = 'off_PLP{:07d}_{:d}.xml'.format(self._rnumber, index)
        d['_r'] = repr(self.m_ref[scanpoint].tolist()).strip(',[]')
        d['_qz'] = repr(self.m_qz[scanpoint].tolist()).strip(',[]')
        d['_dr'] = repr(self.m_ref_sd[scanpoint].tolist()).strip(',[]')
        d['_qy'] = repr(self.m_qy[scanpoint].tolist()).strip(',[]')

        thefile = s.safe_substitute(d)

        g = f
        if not hasattr(f, 'read'):
            own_fh = open(f, 'w')
            g = own_fh

        try:
            g.write(thefile)
            g.truncate()
        finally:
            g.close()

    def _reduce_single_angle(self, scale=1):
        """
        Reduce a single angle.
        """
        n_spectra = self.reflected_beam.n_spectra
        n_tpixels = np.size(self.reflected_beam.m_topandtail, 1)
        n_ypixels = np.size(self.reflected_beam.m_topandtail, 2)

        # calculate omega and two_theta depending on the mode.
        mode = self.reflected_beam.mode

        # we'll need the wavelengths to calculate Q.
        wavelengths = self.reflected_beam.m_lambda
        m_twotheta = np.zeros((n_spectra, n_tpixels, n_ypixels))

        detector_z_difference = (self.reflected_beam.detector_z -
                                 self.direct_beam.detector_z)

        beampos_z_difference = (self.reflected_beam.m_beampos -
                                self.direct_beam.m_beampos)

        total_z_deflection = (detector_z_difference +
                              beampos_z_difference * Y_PIXEL_SPACING)

        if mode in ['FOC', 'POL', 'POLANAL', 'MT']:
            # omega_nom.shape = (N, )
            omega_nom = np.degrees(np.arctan(total_z_deflection /
                                   self.reflected_beam.detector_y) / 2.)

            '''
            Wavelength specific angle of incidence correction
            This involves:
            1) working out the trajectory of the neutrons through the
            collimation system.
            2) where those neutrons intersect the sample.
            3) working out the elevation of the neutrons when they hit the
            sample.
            4) correcting the angle of incidence.
            '''
            speeds = general.wavelength_velocity(wavelengths)
            collimation_distance = self.reflected_beam.cat.collimation_distance
            s2_sample_distance = (self.reflected_beam.cat.sample_distance -
                                  self.reflected_beam.cat.slit2_distance)

            # work out the trajectories of the neutrons for them to pass
            # through the collimation system.
            trajectories = find_trajectory(collimation_distance / 1000.,
                                           0, speeds)

            # work out where the beam hits the sample
            res = parabola_line_intersection_point(s2_sample_distance / 1000,
                                                   0,
                                                   trajectories,
                                                   speeds,
                                                   omega_nom[:, np.newaxis])
            intersect_x, intersect_y, x_prime, elevation = res

            # correct the angle of incidence with a wavelength dependent
            # elevation.
            omega_corrected = omega_nom[:, np.newaxis] - elevation

            m_twotheta += np.arange(n_ypixels * 1.)[np.newaxis, np.newaxis, :]
            m_twotheta -= self.direct_beam.m_beampos[:, np.newaxis, np.newaxis]
            m_twotheta *= Y_PIXEL_SPACING
            m_twotheta += detector_z_difference
            m_twotheta /= (
                self.reflected_beam.detector_y[:, np.newaxis, np.newaxis])
            m_twotheta = np.arctan(m_twotheta)
            m_twotheta = np.degrees(m_twotheta)

            # you may be reflecting upside down, reverse the sign.
            upside_down = np.sign(omega_corrected[:, 0])
            m_twotheta *= upside_down[:, np.newaxis, np.newaxis]
            omega_corrected *= upside_down[:, np.newaxis]

        elif mode in ['SB', 'DB']:
            omega = np.arctan(total_z_deflection /
                              self.reflected_beam.detector_y) / 2.

            m_twotheta += np.arange(n_ypixels * 1.)[np.newaxis, np.newaxis, :]
            m_twotheta -= self.direct_beam.m_beampos[:, :, np.newaxis]
            # m_theta *= Y_PIXEL_SPACING
            m_twotheta += detector_z_difference
            m_twotheta -= (
                self.reflected_beam.detector_y[:, np.newaxis, np.newaxis] *
                np.tan(omega[:, :, np.newaxis]))

            m_twotheta /= (
                self.reflected_beam.detector_y[:, np.newaxis, np.newaxis])
            m_twotheta = np.arctan(m_twotheta)
            m_twotheta += omega[:, :, np.newaxis]

            # still in radians at this point
            omega_corrected = np.degrees(omega)
            m_twotheta = np.degrees(m_twotheta)

        '''
        --Specular Reflectivity--
        Use the (constant wavelength) spectra that have already been integrated
        over 2theta (in processnexus) to calculate the specular reflectivity.
        Beware: this is because m_topandtail has already been divided through
        by monitor counts and error propagated (at the end of processnexus).
        Thus, the 2theta pixels are correlated to some degree. If we use the 2D
        plot to calculate reflectivity
        (sum {Iref_{2theta, lambda}}/I_direct_{lambda}) then the error bars in
        the reflectivity turn out much larger than they should be.
        '''
        ydata, ydata_sd = EP.EPdiv(self.reflected_beam.m_spec,
                                   self.reflected_beam.m_spec_sd,
                                   self.direct_beam.m_spec,
                                   self.direct_beam.m_spec_sd)

        # calculate the 1D Qz values.
        xdata = general.q(omega_corrected, wavelengths)
        xdata_sd = (self.reflected_beam.m_lambda_fwhm /
                    self.reflected_beam.m_lambda) ** 2
        xdata_sd += (self.reflected_beam.domega[:, np.newaxis] /
                     omega_corrected) ** 2
        xdata_sd = np.sqrt(xdata_sd) * xdata

        '''
        ---Offspecular reflectivity---
        normalise the counts in the reflected beam by the direct beam
        spectrum this gives a reflectivity. Also propagate the errors,
        leaving the fractional variance (dr/r)^2.
        --Note-- that adjacent y-pixels (same wavelength) are correlated in
        this treatment, so you can't just sum over them.
        i.e. (c_0 / d) + ... + c_n / d) != (c_0 + ... + c_n) / d
        '''
        m_ref, m_ref_sd = EP.EPdiv(
            self.reflected_beam.m_topandtail,
            self.reflected_beam.m_topandtail_sd,
            self.direct_beam.m_spec[:, :, np.newaxis],
            self.direct_beam.m_spec_sd[:, :, np.newaxis])

        # you may have had divide by zero's.
        m_ref = np.where(np.isinf(m_ref), 0, m_ref)
        m_ref_sd = np.where(np.isinf(m_ref_sd), 0, m_ref_sd)

        # calculate the Q values for the detector pixels.  Each pixel has
        # different 2theta and different wavelength, ASSUME that they have the
        # same angle of incidence
        qx, qy, qz = general.q2(omega_corrected[:, :, np.newaxis],
                                m_twotheta,
                                0,
                                wavelengths[:, :, np.newaxis])

        reduction = {}
        reduction['xdata'] = self.xdata = xdata
        reduction['xdata_sd'] = self.xdata_sd = xdata_sd
        reduction['ydata'] = self.ydata = ydata
        reduction['ydata_sd'] = self.ydata_sd = ydata_sd
        reduction['omega'] = omega_corrected
        reduction['m_ref'] = self.m_ref = m_ref
        reduction['m_ref_sd'] = self.m_ref_sd = m_ref_sd
        reduction['qz'] = self.m_qz = qz
        reduction['qy'] = self.m_qy = qy
        reduction['nspectra'] = self.n_spectra = n_spectra
        reduction['start_time'] = self.reflected_beam.start_time
        reduction['datafile_number'] = self.datafile_number = (
            self.reflected_beam.datafile_number)

        fnames = []
        if self.save:
            datafilename = self.reflected_beam.datafilename
            datafilename = os.path.basename(datafilename.split('.nx.hdf')[0])

            for i in range(n_spectra):
                data_tup = self.data(scanpoint=i)
                dataset = ReflectDataset(data_tup)

                fname = '{0}_{1}.dat'.format(datafilename, i)
                fnames.append(fname)
                with open(fname, 'wb') as f:
                    dataset.save(f)

                fname = '{0}_{1}.xml'.format(datafilename, i)
                with open(fname, 'wb') as f:
                    dataset.save_xml(f,
                                     start_time=reduction['start_time'][i])

        reduction['fname'] = fnames
        return deepcopy(reduction)


def reduce_stitch(reflect_list, direct_list, norm_file_num=None,
                  data_folder=None, trim_trailing=True, save=True, **kwds):
    """
    Reduces a list of reflected beam run numbers and a list of corresponding
    direct beam run numbers from the Platypus reflectometer. If there are
    multiple reflectivity files they are spliced together.

    Parameters
    ----------
    reflect_list : list
        Reflected beam run numbers, e.g. `[708, 709, 710]`
        708 corresponds to the file PLP0000708.nx.hdf.
    direct_list : list
        Direct beam run numbers, e.g. `[711, 711, 711]`
    norm_file_num : int, optional
        The run number for the water flood field correction.
    data_folder : str, optional
        Where is the raw data stored?
    trim_trailing : bool, optional
        When datasets are spliced together do you want to remove points in the
        overlap region from the preceding dataset?
    save : bool, optional
        If `True` then the spliced file is written to a file (in the working
        directory) with a name like: `c_PLP0000708.dat`.
    kwds : dict, optional
        Options passed directly to `refnx.reduce.platypusnexus.process`,
        for processing of individual spectra. Look at that method docstring
        for specification of options.

    Returns
    -------
    combined_dataset, reduced_filename : refnx.dataset.ReflectDataset, str
        The combined dataset and the file name of the reduced data, if it was
        saved. If it wasn't saved `reduced_filename` is `None`.
    """
    scale = kwds.get('scale', 1.)

    # now reduce all the files.
    zipped = zip(reflect_list, direct_list)

    combined_dataset = ReflectDataset()

    if data_folder is None:
        data_folder = os.getcwd()

    if norm_file_num:
        norm_datafile = number_datafile(norm_file_num)
        kwds['h5norm'] = norm_datafile

    for index, val in enumerate(zipped):
        reflect_datafile = os.path.join(data_folder,
                                        number_datafile(val[0]))
        direct_datafile = os.path.join(data_folder,
                                       number_datafile(val[1]))

        reduced = PlatypusReduce(direct_datafile,
                                 reflect=reflect_datafile,
                                 save=save,
                                 **kwds)
        if not index:
            reduced.scale(scale)

        combined_dataset.add_data(reduced.data(),
                                  requires_splice=True,
                                  trim_trailing=trim_trailing)

    fname = None
    if save:
        # this will give us <fname>.nx.hdf
        # if reflect_list was an integer you'll get PLP0000708.nx.hdf
        fname = number_datafile(reflect_list[0])
        # now chop off .nx.hdf extension
        fname = basename_datafile(fname)

        fname_dat = 'c_{0}.dat'.format(fname)
        with open(fname_dat, 'wb') as f:
            combined_dataset.save(f)
        fname_xml = 'c_{0}.xml'.format(fname)
        with open(fname_xml, 'wb') as f:
            combined_dataset.save_xml(f)

    return combined_dataset, fname_dat


if __name__ == "__main__":
    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

    a = reduce_stitch([708, 709, 710], [711, 711, 711], rebin_percent=2)

    a.save('test1.dat')

    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
