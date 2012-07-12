from __future__ import division
import numpy as np
import spectrum
from time import gmtime, strftime
import string

class PlatypusSpectrum(spectrum.Spectrum):

	"""
	
		Objects of this class represent a processed Platypus spectrum
		
			N is the number of scanpoints
			T is the number of wavelength/timebins in the rebinned data.
			Y is the number of y pixels in the rebinned data
			
			The following object attributes are available:
			
				self.basedir
				
				self.datafilename
				
				self.datafilenumber
				
				self.normfilename
				
				self.normfilenumber
			
				self.M_topandtail[N, T, Y]	-rebinned, background subtracted, detector image. There are Y vertical pixels
				
				self.M_topandtailSD[N, T, Y]	-corresponding SD for M_topandtail
				
				self.M_spec[N, T]			-integrated intensity for a particular wavelength bin
				
				self.M_specSD[N, T]			-corresponding SD for M_spec (standard deviation
				
				self.M_lambda[N, T]			-wavelength values for each bin (in Angstrom)
				
				self.M_lambdaSD[N, T]		-uncertainty in M_lambda (FWHM)
				
				self.M_lambdaHIST[N, T+1]	-bin edges for M_lambda
				
				self.M_spectof[N, T]			-time of flight values for each bin
				
				self.mode					-mode for a given acquisition (FOC, MT, DB, SB, etc)
				
				self.detectorZ[N]			-detector vertical translation (mm)
				
				self.detectorY				-detector horizontal translation from the sample (mm)
				
				self.domega					-angular divergence of the beam (radians)
				
				self.M_beampos[N, T]			-beamposition for specular beam
				
				self.lopx = lopx			-lowest pixel to be used for integrating specular beam
				
				self.hipx = hipx			-highest pixel to be used for integrating specular beam
				
				self.title					-title of experiment
				
				self.sample					-sample name
				
				self.user					-user name

	"""
	
	__allowed_attributes = ['M_spec', 'M_specSD', 'M_topandtail', 'M_topandtailSD', 'M_beampos', 'M_lambda', 'M_lambdaSD', 'M_lambdaHIST', 'bmon1_counts',
					'M_spectof', 'mode', 'detectorZ', 'detectorY', 'domega', 'lopx', 'hipx', 'title', 'sample', 'user', 'numspectra',
					'basedir', 'datafilename', 'datafilenumber', 'normfilename', 'normfilenumber']
				
			
	def __init__(self, **kwds):
		super(PlatypusSpectrum, self).__init__()

		for key, value in kwds.items():
			if key in self.__allowed_attributes:
				object.__setattr__(self, key, value)
		
		
	def write_spectrum_XML(self, f, scanpoint = 0):
		"""
			
			This method writes an XML representation of the corrected spectrum to the file f (supplied by callee).
			You must have processed the data before calling this method. The default scanpoint is 0.  See process() for further
			details on what scanpoint means.
		
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
		<dlambda type="FWHM" units="1/A">$dl</dlambda>
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
			
			This method writes an dat representation of the corrected spectrum to the file f (supplied by callee).
			The default scanpoint is 0.  See ProcessPlatypusNexus.process() for further
			details on what scanpoint means.
		
		"""
		
		for L, I, dI, dL in zip(self.M_lambda[scanpoint], self.M_spec[scanpoint], self.M_specSD[scanpoint], self.M_lambdaSD[scanpoint]):
			thedata = '{:g}\t{:g}\t{:g}\t{:g}\n'.format(L, I, dI, dL)
			f.write(thedata)
		
		f.truncate()
		
		return True

