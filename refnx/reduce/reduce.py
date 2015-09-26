from __future__ import division
import numpy as np
import platypusnexus as pn
import refnx.util.ErrorProp as EP
import refnx.util.general as general
import string
from time import gmtime, strftime
import refnx.reduce.parabolic_motion as pm
from refnx.dataset import reflectdataset


class ReducePlatypus(object):
    """
    Reduces Platypus reflectometer data to give the specular reflectivity.
    Offspecular data maps are also produced.
    """

    def __init__(self, h5ref, h5direct, scale=1., **kwds):
        """
        Parameters
        ----------
        h5ref: string or file-like object
            A string containing the path to the specularly reflected hdf5 file,
            or the hdf5 file itself.
        h5direct: string or file-like object
            A string containing the path to the direct beam hdf5 file,
            or the hdf5 file itself.
        scale: float
            Divide all the reflectivity values by this number.

        kwds: dict
            Options passed directly to `refnx.reduce.platypusnexus.process`,
            for processing of individual spectra. Look at that method docstring
            for specification of options.

        Returns
        -------

        Notes
        -----
        Following successful construction of the object the following
        attributes are available:
        self.x[N, T]:
            Q values
        self.x_sd[N, T]:
            uncertainty in Q values (FWHM)
        self.y[N, T]:
            specular reflectivity
        self.y_sd[N, T]
            uncertainty in specular reflectivity (SD)
        self.m_ref[N, T, Y]
            offspecular reflectivity map
        self.m_refSD[N, T, Y]			
            uncertainty in offspecular reflectivity
        self.m_qz[N, T, Y]
            Qz for offspecular map
        self.m_qy[N, T, Y]
            Qy for offspecular map
        self.n_spectra
            N
        self.datafile_number
            run number for the reflected beam
        self.reflected_beam
            a platypusnexus.PlatypusNexus object for the reflected beam
            spectrum
        self.direct_beam
            a platypusnexus.PlatypusNexus object for the direct beam
            spectrum

        N corresponds to the number of spectra
        T corresponds to the number of Q (wavelength) bins
        Y corresponds to the number of y pixels on the detector.

        This class reduces all the spectra present in the reflected beam file
        (see platypusnexus.PlatypusNexus.process for eventmode
        specification and other related options), but aggregates all data in
        the direct beam spectrum.

        """
        keywords = kwds.copy()
        keywords['direct'] = False

        # get the reflected beam spectrum
        self.reflected_beam = pn.PlatypusNexus(h5ref)
        self.reflected_beam.process(**keywords)

        # get the direct beam spectrum
        keywords['direct'] = True
        keywords['integrate'] = -1

        # got to use the same wavelength bins as the reflected spectrum.
        keywords['wavelength_bins'] = self.reflected_beam.m_lambda_hist[0]

        self.direct_beam = pn.PlatypusNexus(h5direct)
        self.direct_beam.process(**keywords)

        self.__reduce_single_angle()
        self.scale(scale)

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
        __template_ref_xml = """<?xml version="1.0"?>
        <REFroot xmlns="">
        <REFentry time="$time">
        <Title>$title</Title>
        <User>$user</User>
        <REFsample>
        <ID>$sample</ID>
        </REFsample>
        <REFdata axes="Qz:Qy" rank="2" type="POINT" spin="UNPOLARISED" dim="$_numpointsz:$_numpointsy">
        <Run filename="$_rnumber" preset="" size="">
        </Run>
        <R uncertainty="dR">$_r</R>
        <Qz uncertainty="dQz" units="1/A">$_qz</Qz>
        <dR type="SD">$_dr</dR>
        <Qy type="_FWHM" units="1/A">$_qy</Qy>
        </REFdata>
        </REFentry>
        </REFroot>"""
        d = dict()
        d['time'] = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        d['_rnumber'] = self.reflected_beam.datafilenumber
        d['_numpointsz'] = np.size(self.m_ref, 1)
        d['_numpointsy'] = np.size(self.m_ref, 2)

        s = string.Template(__template_ref_xml)

        # filename = 'off_PLP{:07d}_{:d}.xml'.format(self._rnumber, index)
        d['_r'] = string.translate(repr(self.m_ref[scanpoint].tolist()), None, ',[]')
        d['_qz'] = string.translate(repr(self.m_qz[scanpoint].tolist()), None, ',[]')
        d['_dr'] = string.translate(repr(self.m_ref_sd[scanpoint].tolist()), None, ',[]')
        d['_qy'] = string.translate(repr(self.m_qy[scanpoint].tolist()), None, ',[]')

        thefile = s.safe_substitute(d)
        f.write(thefile)
        f.truncate()

    def __reduce_single_angle(self):
        n_spectra = self.reflected_beam.n_spectra
        n_tpixels = np.size(self.reflected_beam.m_topandtail, 1)
        n_ypixels = np.size(self.reflected_beam.m_topandtail, 2)

        # calculate omega and two_theta depending on the mode.
        mode = self.reflected_beam.mode

        # we'll need the wavelengths to calculate Q.
        wavelengths = self.reflected_beam.m_lambda
        m_twotheta = np.zeros((n_spectra, n_tpixels, n_ypixels))

        if mode in ['FOC', 'POL', 'POLANAL', 'MT']:
            detector_z_difference = (self.reflected_beam.detector_z -
                                     self.direct_beam.detector_z)
            beampos_z_difference = (self.reflected_beam.m_beampos
                                    - self.direct_beam.m_beampos)

            total_z_deflection = (detector_z_difference
                                  + beampos_z_difference * pn.Y_PIXEL_SPACING)

            # omega_nom.shape = (N, )
            omega_nom = np.degrees(np.arctan(total_z_deflection
                                   / self.reflected_beam.detector_y) / 2.)

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
            s2_sample_distance = (self.reflected_beam.cat.sample_distance
                                  - self.reflected_beam.cat.slit2_distance)

            # work out the trajectories of the neutrons for them to pass
            # through the collimation system.
            trajectories = pm.find_trajectory(collimation_distance / 1000.,
                                              0, speeds)
            
            # work out where the beam hits the sample
            res = pm.parabola_line_intersection_point(s2_sample_distance / 1000,
                                                      0,
                                                      trajectories,
                                                      speeds,
                                                      omega_nom[:, np.newaxis])
            intersect_x, intersect_y, x_prime, elevation = res

            # correct the angle of incidence with a wavelength dependent
            # elevation.
            omega_corrected = omega_nom[:, np.newaxis] - elevation

        elif mode == 'SB' or mode == 'DB':
            omega = self.reflected_beam.M_beampos + self.reflected_beam.detectorZ[:, np.newaxis]
            omega -= self.direct_beam.M_beampos + self.direct_beam.detectorZ
            omega /= 2 * self.reflected_beam.detectorY[:, np.newaxis, np.newaxis]
            omega = np.arctan(omega)

            m_twotheta += np.arange(n_ypixels * 1.)[np.newaxis, np.newaxis, :] * pn.Y_PIXEL_SPACING
            m_twotheta += self.reflected_beam.detectorZ[:, np.newaxis, np.newaxis]
            m_twotheta -= self.direct_beam.M_beampos[:, :, np.newaxis] + self.direct_beam.detectorZ
            m_twotheta -= self.reflected_beam.detectorY[:, np.newaxis, np.newaxis] * np.tan(omega[:, :, np.newaxis])

            m_twotheta /= self.reflected_beam.detectorY[:, np.newaxis, np.newaxis]
            m_twotheta = np.arctan(m_twotheta)
            m_twotheta += omega[:, :, np.newaxis]

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
        xdata_sd = (self.reflected_beam.m_lambda_fwhm
                    / self.reflected_beam.m_lambda) ** 2
        xdata_sd += (self.reflected_beam.domega[:, np.newaxis]
                     / omega_corrected) ** 2
        xdata_sd = np.sqrt(xdata_sd) * xdata

        '''
        ---Offspecular reflectivity---
        normalise the counts in the reflected beam by the direct beam
        spectrum this gives a reflectivity. Also propagate the errors,
        leaving the fractional variance (dr/r)^2.
        --Note-- that adjacent y-pixels (same wavelength) are correlated in this
        treatment, so you can't just sum over them.
        i.e. (c_0 / d) + ... + c_n / d) != (c_0 + ... + c_n) / d
        '''
        m_ref, m_ref_sd = EP.EPdiv(self.reflected_beam.m_topandtail,
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

        self.xdata = xdata
        self.xdata_sd = xdata_sd
        self.ydata = ydata
        self.ydata_sd = ydata_sd
        self.m_ref = m_ref
        self.m_ref_sd = m_ref_sd
        self.m_qz = qz
        self.m_qy = qy
        self.n_spectra = n_spectra
        self.datafile_number = self.reflected_beam.datafile_number


def reduce_stitch_files(reflect_list, direct_list, norm_file_num=None,
                        trim_trailing=True, **kwds):
    """
    Reduces a list of reflected beam run numbers and a list of corresponding
    direct beam run numbers from the Platypus reflectometer.
    e.g.
        reflect_list = [708, 709, 710]
        direct_list = [711, 711, 711]

    708 is corresponds to the file PLP0000708.nx.hdf.

    norm_file_num is the run number for the water flood field correction.

    kwds : dict, optional
        Options passed directly to `refnx.reduce.platypusnexus.process`,
        for processing of individual spectra. Look at that method docstring
        for specification of options.
    """
    scale = kwds.get('scale', 1.)

    # now reduce all the files.
    zipped = zip(reflect_list, direct_list)

    combined_dataset = reflectdataset.ReflectDataset()

    if norm_file_num:
        norm_datafile = pn.number_datafile(norm_file_num)
        kwds['h5norm'] = norm_datafile

    for index, val in enumerate(zipped):
        reflect_datafile = pn.number_datafile(val[0])
        direct_datafile = pn.number_datafile(val[1])

        reduced = ReducePlatypus(reflect_datafile, direct_datafile,
                                 **kwds)
        if not index:
            reduced.scale(scale)

        combined_dataset.add_data(reduced.data(), requires_splice=True,
                                  trim_trailing=trim_trailing)

    return combined_dataset


if __name__ == "__main__":
    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

    a = reduce_stitch_files([708, 709, 710], [711, 711, 711], rebin_percent=2)

    a.save('test1.dat')

    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

