from __future__ import division
import numpy as np
import spectrum
from time import gmtime, strftime
import string

class PlatypusSpectrum(object):

	"""
	Representation of a processed Platypus spectrum

    N is the number of scanpoints
    T is the number of wavelength/timebins in the rebinned data.
    Y is the number of y pixels in the rebinned data

    The following object attributes are available:
    path : str
        path to the dataset
    datafilename : str
        datafile name
    datafilenumber : int
        number of datafile
    normfilename
    normfilenumber
    M_topandtail : np.ndarray
        Rebinned, Background subtracted, Detector image. [N, T, Y]
    M_topandtailSD : np.ndarray
        Corresponding standard deviation for M_topandtail
    M_spec : np.ndarray
    	Specular intensity [N, T]
    M_lambda : np.ndarray
        Wavelength values [N, T]
    M_lambdaSD : np.ndarray
    	Uncertainty in M_lambda (_FWHM)
    M_lambdaHIST : np.ndarray
        Bin edges for M_lambda [N, T + 1]
    M_spectof : np.ndarray
        Time of flight values
    mode : str
    	Mode for a given acquisition (FOC, MT, DB, SB, etc)
    detectorZ : np.ndarray
    	Detector vertical translation (mm)
    detectorY : np.ndarray
    	Detector horizontal translation from the sample (mm)
    domega : np.ndarray
    	Angular divergence of the beam (radians)
    M_beampos : np.ndarray
        Beamposition for specular beam [N, T]
    lopx : float
        Lowest pixel to be used for integrating specular beam
    hipx : float
        Highest pixel to be used for integrating specular beam
    title : str
    	Title of experiment
    sample : str
    	Sample name
    user : str
        User name
	"""
	
	__allowed_attributes = ['M_spec', 'M_spec_sd', 'M_topandtail',
                            'M_topandtail_sd', 'M_beampos', 'M_lambda',
                            'M_lambda_sd', 'M_lambda_hist', 'bm1_counts',
                            'M_spec_tof', 'mode', 'detectorZ', 'detectorY',
                            'domega', 'lopx', 'hipx', 'title', 'sample',
                            'user', 'num_spectra', 'path', 'datafilename',
                            'datafile_number', 'normfilename',
                            'normfilenumber']
				
			
	def __init__(self, **kwds):
		super(PlatypusSpectrum, self).__init__()

		for key, value in kwds.items():
			if key in self.__allowed_attributes:
				object.__setattr__(self, key, value)
		
		
	def write_spectrum_XML(self, f, scanpoint=0):
		"""
        This method writes an XML representation of the corrected spectrum to
        file.

        Parameters
        ----------
        f : file-like object or str
            The file to write the spectrum to
        scanpoint : int
            Which scanpoint to write.
		"""
		
		spectrum_template = """<?xml version="1.0"?>
		<REFroot xmlns="">
		<REFentry time="$time">
		<Title>$title</Title>
		<REFdata axes="lambda" rank="1" type="POINT" spin="UNPOLARISED" dim="$numpoints">
		<Run filename="$runnumber"/>
		<R uncertainty="dR">$r</R>
		<lambda uncertainty="dlambda" units="1/A">$l</lambda>
		<dR type="SD">$dr</dR>
		<dlambda type="_FWHM" units="1/A">$dl</dlambda>
		</REFdata>
		</REFentry>
		</REFroot>"""
		
		
		s = string.Template(spectrum_template)
		d = dict()
		d['title'] = self.title
		d['time'] = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
		#sort the data
		sorted = np.argsort(self.M_lambda[0])
		r = self.M_spec[:,sorted]
		l = self.M_lambda[:, sorted]
		dl = self.M_lambdaSD[:, sorted]
		dr = self.M_specSD[:, sorted]
		d['numpoints'] = np.size(r, axis = 1)
		#filename = 'PLP{:07d}_{:d}.spectrum'.format(self.datafilenumber, index)
		d['runnumber'] = 'PLP{:07d}'.format(self.datafilenumber)
			
		d['r'] = string.translate(repr(r[scanpoint].tolist()), None, ',[]')
		d['dr'] = string.translate(repr(dr[scanpoint].tolist()), None, ',[]')
		d['l'] = string.translate(repr(l[scanpoint].tolist()), None, ',[]')
		d['dl'] = string.translate(repr(dl[scanpoint].tolist()), None, ',[]')
		thefile = s.safe_substitute(d)
		f.write(thefile)
		f.truncate()
		
		return True
		
	def write_spectrum_dat(self, f, scanpoint = 0):
		"""
        This method writes a dat representation of the corrected spectrum to
        file.

        Parameters
        ----------
        f : file-like object or str
            The file to write the spectrum to
        scanpoint : int
            Which scanpoint to write.
		"""
		for L, I, dI, dL in zip(self.M_lambda[scanpoint],
                                self.M_spec[scanpoint],
                                self.M_specSD[scanpoint],
                                self.M_lambdaSD[scanpoint]):

			thedata = '{:g}\t{:g}\t{:g}\t{:g}\n'.format(L, I, dI, dL)
			f.write(thedata)
		
		f.truncate()
		
		return True

