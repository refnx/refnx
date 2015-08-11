from __future__ import division
import numpy as np
import h5py
import platypusnexus as pn
import refnx.util.ErrorProp as EP
import refnx.util.general as general
import string
from time import gmtime, strftime
import refnx.reduce.parabolic_motion as pm
import os
    
class ReducePlatypus(object):
    """
    Reduces Platypus reflectometer data to give the specular reflectivity.
    Offspecular data maps are also produced.
    """

    def __init__(self, h5ref, h5direct, **kwds):
        """
        Parameters
        ----------
        h5ref: string or file-like object
            A string containing the path to the specularly reflected hdf5 file,
            or the hdf5 file itself.
        h5direct: string or file-like object
            A string containing the path to the direct beam hdf5 file,
            or the hdf5 file itself.

        kwds: dict
            Options passed directly to refnx.reduce.platypusnexus.process, look
            at that method docstring for specification of options.

        Returns
        -------


        Notes
        -----
        Following successful construction of the object the following
        attributes are available:
        self.xdata[N, T]:
            Q values
        self.xdata_sd[N, T]:
            uncertainty in Q values (FWHM)
        self.ydata[N, T]:
            specular reflectivity
        self.ydata_sd[N, T]
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
        (see platypusnexus.PlatypusNexus.process nexus for eventmode
        specification and other related options), but aggregates all data in
        the direct beam spectrum.

        """
        keywords = kwds.copy()
        keywords['direct'] = False

        # get the reflected beam spectrum
        self.reflected_beam = pn.PlatypusNexus(h5ref)
        self.reflected_beam.process(**keywords)

        # get the reflected beam spectrum
        keywords['direct'] = True
        keywords['integrate'] = -1
        self.direct_beam = pn.PlatypusNexus(h5direct)
        self.direct_beam.process(**keywords)

        self.__reduce_single_angle()

    def data(self, scanpoint=0):
        """
        The specular reflectivity
        
        Parameters
        ----------
        scanpoint: int
            Find a particular specular reflectivity image. scanpoints upto
            `self.n_spectra - 1` can be specified.

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

    def get_reflected_dataset(self, scanpoint=0):
        """

            returns a reflectdataset.ReflectDataset() created from this Reduce object. By default a scanpoint of 0 is
            used. scanpoints upto self.numspectra - 1 can be specified.

        """
        reflectedDatasetObj = reflectdataset.ReflectDataset([self], scanpoint = scanpoint)
#		reflectedDatasetObj.add_dataset(self, scanpoint = scanpoint)
        return reflectedDatasetObj

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

        #			filename = 'off_PLP{:07d}_{:d}.xml'.format(self._rnumber, index)
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

        if mode == 'FOC' or mode == 'POL' or mode == 'POLANAL' or mode == 'MT':
            detector_z_difference = (self.reflected_beam.detector_z -
                                     self.direct_beam.detector_z)
            beampos_z_difference = (self.reflected_beam.m_beampos
                                    - self.direct_beam.m_beampos)

            total_z_deflection = (detector_z_difference
                                  + beampos_z_difference * pn.Y_PIXEL_SPACING)

            # omega_nom.shape = (N, )
            omega_nom = np.arctan(total_z_deflection
                                  / self.reflected_beam.detector_y)

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
            trajectories = pm.find_trajectory(collimation_distance, 0, speeds)
            
            # work out where the beam hits the sample
            res = pm.parabola_line_intersection_point(s2_sample_distance,
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
        Calculate by dividing the two integrated spectra, rather than operating
        on 2D image.
        '''
        ydata, ydata_sd = EP.EPdiv(self.reflected_beam.m_spec,
                                   self.reflected_beam.m_spec_sd,
                                   self.direct_beam.m_spec,
                                   self.direct_beam.m_spec_sd)

        # calculate the 1D Qz values.
        xdata = general.q(omega_corrected, wavelengths)
        xdata_sd = (self.reflected_beam.m_lambda_sd / self.reflected_beam.m_lambda)**2
        xdata_sd += (self.reflected_beam.domega[:, np.newaxis] / omega_corrected) ** 2
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
        self.datafile_number = self.reflected_beam.datafilenumber


def sanitize_string_input(file_list_string):
    """
    
        given a string like '1 2 3 4 1000 -1 sijsiojsoij' return an integer list where the numbers are greater than 0 and less than 9999999
    it strips the string.ascii_letters and any string.punctuation, and converts all the numbers to ints.
    
    """
    temp = [x.translate(None, string.punctuation).translate(None, string.ascii_letters).split() for x in file_list_string]
    return [int(item) for sublist in temp for item in sublist if 0 < int(item) < 9999999]


def reduce_stitch_files(reflect_list, direct_list, normfilenumber = None, **kwds):
    """

        reduces a list of reflected beam run numbers and a list of corresponding direct beam run numbers from the Platypus reflectometer.
        e.g.
            reflect_list = [708, 709, 710]
            direct_list = [711, 711, 711]

        708 is corresponds to the file PLP0000708.nx.hdf.

        normfilenumber is the run number for the water flood field correction.

        kwds is passed onto processplatypusnexus.ProcessPlatypusNexus.process, look at that docstring for specification of options.

    """

    scalefactor = kwds.get('scalefactor', 1.)

    #now reduce all the files.
    zipped = zip(reflect_list, direct_list)

    combineddataset = reflectdataset.ReflectDataset()

    if kwds.get('basedir'):
        basedir = kwds.get('basedir')
    else:
        kwds['basedir'] = os.getcwd()
        basedir = os.getcwd()

    normfiledatafilename = ''
    if normfilenumber:
        nfdfn = 'PLP{0:07d}.nx.hdf'.format(int(abs(normfilenumber)))
        for root, dirs, files in os.walk(self.basedir):
            if nfdfn in files:
                normfiledatafilename = os.path.join(root, nfdfn)
                break

    for index, val in enumerate(zipped):
        rdfn = 'PLP{0:07d}.nx.hdf'.format(int(abs(val[0])))
        ddfn = 'PLP{0:07d}.nx.hdf'.format(int(abs(val[1])))
        reflectdatafilename = ''
        directdatafilename = ''

        for root, dirs, files in os.walk(basedir):
            if rdfn in files:
                reflectdatafilename = os.path.join(root, rdfn)
            if ddfn in files:
                directdatafilename = os.path.join(root, ddfn)
            if len(reflectdatafilename) and len(directdatafilename):
                break

        with h5py.File(reflectdatafilename, 'r') as h5ref, h5py.File(directdatafilename, 'r') as h5direct:
            if len(normfiledatafilename):
                with h5py.File(normfiledatafilename, 'r') as h5norm:
                    reduced = Reduce(h5ref, h5direct, h5norm = h5norm, **kwds)
            else:
                reduced = Reduce(h5ref, h5direct, **kwds)

        combineddataset.add_dataset(reduced)

    return combineddataset


if __name__ == "__main__":
    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

    a = reduce_stitch_files([708, 709, 710], [711,711,711])

    a.rebin(rebinpercent = 4)
    with open('test.xml', 'w') as f:
        a.save(f)

    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

