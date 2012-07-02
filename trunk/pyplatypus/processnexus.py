from __future__ import division
import numpy as np
import h5py as h5
import  ErrorProp as EP
import utility as ut
import Qtransforms as qtrans
from scipy.optimize import curve_fit
from scipy.stats import t
import rebin
import string
import struct
from array import array
from time import gmtime, strftime
import os
import argparse

Y_PIXEL_SPACING = 1.177    #in mm
O_C1 = 1.04719755
O_C2 = 0.17453293
O_C3 = 0.43633231
O_C4 = 1.04719755
O_C1d = 60.
O_C2d = 10.
O_C3d = 25.
O_C4d = 60.
DISCRADIUS = 350.

#default distances.  These should normally be read from the NeXUS file.
C_CHOPPER1_DISTANCE = 0.
C_CHOPPER2_DISTANCE = 103.
C_CHOPPER3_DISTANCE = 359.
C_CHOPPER4_DISTANCE = 808.
C_SLIT2_DISTANCE = 1909.9
C_SLIT3_DISTANCE = 4767.9
C_GUIDE1_DISTANCE = np.nan
C_GUIDE2_DISTANCE = np.nan
C_SAMPLE_DISTANCE = 5045.4

#the constants below may change frequently
CHOPFREQ = 20                #Hz
ROUGH_BEAM_POSITION = 150        #Rough direct beam position
ROUGH_BEAM_WIDTH = 10
CHOPPAIRING = 3	

class processnexus(object):

	def __init__(self, datafilenumber, **kwds):
		self.datafilenumber = datafilenumber
		self.isprocessed = 0
		self.h5norm = None
		self.basedir = os.getcwd()
		self.detector = None
		
		normfilenumber = kwds.get('normfilenumber', None)

		self.datafilename = 'PLP{0:07d}.nx.hdf'.format(int(abs(self.datafilenumber)))
		if kwds.get('basedir'):
			self.basedir = kwds.get('basedir')
			for root, dirs, files in os.walk(kwds['basedir']):
				if self.datafilename in files:
					self.datafilename = os.path.join(root, self.datafilename)
					break
		
		if normfilenumber:
			self.normfilenumber = normfilenumber
			self.normfilename = 'PLP{0:07d}.nx.hdf'.format(int(abs(self.normfilenumber)))
			
			if kwds.get('basedir'):
				for root, dirs, files in os.walk(kwds['basedir']):
					if self.normfilename in files:
						self.normfilename = os.path.join(root, self.normfilename)
						break
			if self.normfilename:
				self.h5norm = h5.File(self.normfilename, 'r')
				self.h5norm.close()
		
		self.__nexusOpen()
		self.__nexusClose()
				
	def __del__(self):
		self.__nexusClose()
		if self.h5norm:
			self.h5norm.close()


	def process(self, **kwds):
		self.lolambda = kwds.get('lolambda', 2.8)
		self.hilambda = kwds.get('hilambda', 18.)
		self.background = kwds.get('background', True)
		self.eventstreaming = kwds.get('eventstreaming', {'scanpoint':0, 'frame_bins':None})
		self.isdirect = kwds.get('isdirect', False)
		self.peak_pos = kwds.get('peak_pos', None)
		self.typeofintegration = kwds.get('typeofintegration', 0)
		self.expected_width = kwds.get('expected_width', 10.)
		self.omega = kwds.get('omega', 0.)
		self.two_theta = kwds.get('two_theta', 0.) 
		self.rebinpercent = kwds.get('rebinpercent', 4.)
		self.wavelengthbins = kwds.get('wavelengthbins', None)
		self.bmon1_normalise = kwds.get('bmon1_normalise', True) 
		self.verbose = kwds.get('verbose', False) 
	
		self.__nexusOpen()

		self.frequency = self.h5data['entry1/instrument/disk_chopper/ch1speed'][0]
		
		scanpoint = 0
		
		#beam monitor counts for normalising data
		self.bmon1_counts = np.zeros(dtype = 'float64', shape = self.h5data['entry1/monitor/bm1_counts'].shape)
		self.bmon1_counts = self.h5data['entry1/monitor/bm1_counts'][:]
		
		#set up the RAW TOF bins (n, t + 1)
		TOF = np.zeros(dtype = 'float64', shape = self.h5data['entry1/data/time_of_flight'].shape)
		TOF = self.h5data['entry1/data/time_of_flight'][:]

		#event streaming.
		if 'eventstreaming' in kwds:
			scanpoint = self.eventstreaming['scanpoint']
			frame_bins, self.detector, self.bmon1_counts = self.processEventStream(scanpoint = self.eventstreaming['scanpoint'], frame_bins = self.eventstreaming['frame_bins'])
			self.bmon1_counts = np.array((self.bmon1_counts), dtype = 'float64')

			self.numspectra = len(self.detector)
		else:
			#detector(n, t, y, x)    
			self.detector = np.zeros(dtype='int32', shape = self.h5data['entry1/data/hmm'].shape)
			self.detector = self.h5data['entry1/data/hmm'][:]
			
			#you average over the individual measurements in the file
			if self.typeofintegration == 0:
				self.numspectra = 1
				self.detector = np.sum(self.detector, axis = 0)
				self.detector = np.resize(self.detector, (1, np.size(self.detector, 0), np.size(self.detector, 1), np.size(self.detector, 2)))
				self.bmon1_counts = np.array([np.sum(self.bmon1_counts)], dtype = 'float64')
			else:
				self.numspectra = len(self.detector)

		#pre-average over x, leaving (n, t, y) also convert to dp
		self.detector = np.sum(self.detector, axis = 3, dtype = 'float64')

		#detector shape should now be (n, t, y)
		#create the SD of the array
		detectorSD = np.sqrt(self.detector + 1)
		
		#detector normalisation with a water file
		if self.h5norm:
			xbins = h5data['entry1/data/x_bin']
			#shape (y,)
			M_detectornorm, M_detectornormSD = self.__createdetectornorm(xbins[0], xbins[1])
			#detector has shape (n,t,y), shape of M_waternorm should broadcast to (1,1,y)
			self.detector, detectorSD = EP.EPdiv(self.detector, detectorSD, M_detectornorm, M_detectornormSD)
					
		#shape of these is (numspectra, TOFbins)
		M_specTOFHIST = np.zeros((self.numspectra, len(TOF)), dtype = 'float64')
		M_lambdaHIST = np.zeros((self.numspectra, len(TOF)), dtype = 'float64')
		M_specTOFHIST[:] = TOF

		#chopper to detector distances
		#note that if eventstreaming is specified the numspectra is NOT
		#equal to the number of entries in e.g. /longitudinal_translation
		#this means you have to copy values in from the same scanpoint
		chod = np.zeros(self.numspectra, dtype = 'float64')
		detpositions = np.zeros(self.numspectra, dtype = 'float64')
		
		#domega, the angular divergence of the instrument
		domega = np.zeros(self.numspectra, dtype = 'float64')
		
		#process each of the spectra taken in the detector image
		originalscanpoint = scanpoint
		for index in xrange(self.numspectra):
			if self.verbose:
				print datafilenumber, ': processing image for tof params: ', index
			omega = self.h5data['entry1/instrument/parameters/omega'][scanpoint]
			two_theta = self.h5data['entry1/instrument/parameters/twotheta'][scanpoint]
			frequency = self.h5data['entry1/instrument/disk_chopper/ch1speed']
			ch2speed = self.h5data['entry1/instrument/disk_chopper/ch2speed']
			ch3speed = self.h5data['entry1/instrument/disk_chopper/ch3speed']
			ch4speed = self.h5data['entry1/instrument/disk_chopper/ch4speed']
			ch2phase = self.h5data['entry1/instrument/disk_chopper/ch2phase']
			ch3phase = self.h5data['entry1/instrument/disk_chopper/ch3phase']
			ch4phase = self.h5data['entry1/instrument/disk_chopper/ch4phase']
			ch1phaseoffset = self.h5data['entry1/instrument/parameters/chopper1_phase_offset']
			ch2phaseoffset = self.h5data['entry1/instrument/parameters/chopper2_phase_offset']
			ch3phaseoffset = self.h5data['entry1/instrument/parameters/chopper3_phase_offset']
			ch4phaseoffset = self.h5data['entry1/instrument/parameters/chopper4_phase_offset']
			chopper1_distance = self.h5data['entry1/instrument/parameters/chopper1_distance']
			chopper2_distance = self.h5data['entry1/instrument/parameters/chopper2_distance']
			chopper3_distance = self.h5data['entry1/instrument/parameters/chopper3_distance']
			chopper4_distance = self.h5data['entry1/instrument/parameters/chopper4_distance']        
			ss2vg = self.h5data['entry1/instrument/slits/second/vertical/gap']
			ss3vg = self.h5data['entry1/instrument/slits/third/vertical/gap']
			slit2_distance = self.h5data['/entry1/instrument/parameters/slit2_distance']
			slit3_distance = self.h5data['/entry1/instrument/parameters/slit3_distance']
			detectorpos = self.h5data['entry1/instrument/detector/longitudinal_translation']

			pairing = 0
			phaseangle = 0
			master = -1
			slave = -1
			D_CX = 0
			MASTER_OPENING = 0
			freq = frequency[scanpoint] / 60.
			
			#calculate the angular divergence
			domega[index] = 0.68 * np.sqrt((ss2vg[scanpoint]**2 + ss3vg[scanpoint]**2)/((slit3_distance[0] - slit2_distance[0])**2))
		
			#perhaps you've swapped the encoder discs around and you want to use a different pairing
			#there will be slave, master parameters, read the pairing from them.
			#this is because the hardware readout won't match what you actually used.
			#these slave and master parameters need to be set manually.    
			if 'entry1/instrument/parameters/slave' in self.h5data and 'entry1/instrument/parameters/master' in self.h5data:
				master = self.h5data['entry1/instrument/parameters/master'][scanpoint]
				slave = self.h5data['entry1/instrument/parameters/slave'][scanpoint]
				pairing = pairing | 2**slave
				pairing = pairing | 2**master
				if master == 1:
					D_CX = - chopper1_distance[0]
					phaseangle += 0.5 * O_C1d
					MASTER_OPENING = O_C1
				elif master == 2:
					D_CX = - chopper2_distance[0]
					phaseangle += 0.5 * O_C2d
					MASTER_OPENING = O_C2
				elif master == 3:
					D_CX = - chopper3_distance[0]
					phaseangle += 0.5 * O_C3d
					MASTER_OPENING = O_C3
				
				if slave == 2:
					D_CX += chopper2_distance[0]
					phaseangle += 0.5 * O_C2d
					phaseangle += -ch2phase[0] - ch2phaseoffset[0]
				elif slave == 3:
					D_CX += chopper3_distance[0]
					phaseangle += 0.5 * O_C3d
					phaseangle += -ch3phase[0] - ch3phaseoffset[0]
				elif slave == 4:
					D_CX += chopper4_distance[0]
					phaseangle += 0.5 * O_C4d
					phaseangle += ch4phase[0] - ch4phaseoffset[0]
			else:
				#the slave and master parameters don't exist, work out the pairing assuming 1 is the master disk.
				pairing = pairing | 2**1
				MASTER_OPENING = O_C1
				if abs(ch2speed[scanpoint]) > 10:
					D_CX = chopper2_distance[0]
					pairing = pairing | 2**2
					phaseangle = -ch2phase[0] - ch2phaseoffset[0] + 0.5*(O_C2d + O_C1d)
				elif abs(ch3speed[scanpoint]) > 10:
					D_CX = chopper3_distance[0]
					pairing = pairing | 2**3
					phaseangle = -ch3phase[0] - ch3phaseoffset[0] + 0.5*(O_C3d + O_C1d)
				else:
					D_CX = chopper4_distance[0]
					pairing = pairing | 2**4
					phaseangle = ch4phase[0] - ch4phaseoffset[0] + 0.5*(O_C4d + O_C1d)
		
			#work out the total flight length
			chod[index] = self.__chodcalculator(omega, two_theta, pairing, scanpoint)
		
			# toffset - the time difference between the magnet pickup on the choppers (TTL pulse),
			# which is situated in the middle of the chopper window, and the trailing edge of chopper 1, which 
			# is supposed to be time0.  However, if there is a phase opening this time offset has to be 
			# relocated slightly, as time0 is not at the trailing edge.
			poff = ch1phaseoffset[0]
			poffset = 1.e6 * poff/(2. * 360. * freq)
			toffset = poffset + (1.e6 * MASTER_OPENING/2/(2 * np.pi)/freq) - (1.e6 * phaseangle /(360 * 2 * freq))
			M_specTOFHIST[index] -= toffset
			
			detpositions[index] = detectorpos[scanpoint]
			
			if self.eventstreaming:
				M_specTOFHIST[:] = TOF - toffset
				chod[:] = chod[0]
				detpositions[:] = detpositions[0]
				break
			else:
				scanpoint += 1
		
		scanpoint = originalscanpoint
			
		#convert TOF to lambda
		#M_specTOFHIST (n, t) and chod is (n,)
		M_lambdaHIST = qtrans.tof_to_lambda(M_specTOFHIST, chod[:, np.newaxis])
		M_lambda = 0.5 * (M_lambdaHIST[:,1:] + M_lambdaHIST[:,:-1])
		TOF -= toffset

		assert not np.isnan(detectorSD).any()
		assert not np.less(detectorSD, 0).any()
	
		#get the specular ridge on the averaged detector image
		if self.peak_pos:
			beam_centre, beam_SD = peak_pos
		else:
			startingoffset = np.searchsorted(M_lambdaHIST[0], self.hilambda)
			beam_centre, beam_SD = findspecularridge(self.detector, 500)
			if self.verbose:
				print datafilenumber, ": BEAM_CENTRE", datafilenumber, beam_centre

		#TODO gravity correction if direct beam
		if self.isdirect:
			self.detector, detectorSD, M_gravcorrcoefs = correct_for_gravity(self.detector, detectorSD, M_lambda, 0, 2.8, 18)
			beam_centre, beam_SD = findspecularridge(self.detector)
			
		#rebinning in lambda for all detector
		#rebinning is the default option, but sometimes you don't want to.
		#detector shape input is (n, t, y)
		#we want to rebin t.
		if 'wavelengthbins' in kwds:
			rebinning = kwds['wavelengthbins'][0]
		elif 0 < self.rebinpercent < 10.:
			frac = 1. + (self.rebinpercent/100.)
			lowl = (2 * self.lolambda) / ( 1. + frac)
			hil =  frac * (2 * self.hilambda) / ( 1. + frac)
				   
			numsteps = np.floor(np.log10(hil / lowl ) / np.log10(frac)) + 1
			rebinning = np.logspace(np.log10(lowl), np.log10(hil), num = numsteps)
			
		else:
			rebinning = M_lambdaHIST[0,:]
			todel = int(np.interp(self.lolambda, rebinning, np.arange(len(rebinning))))
			rebinning = rebinning[todel:]

			todel = int(np.interp(self.hilambda, rebinning, np.arange(len(rebinning))))
			rebinning = rebinning[0: todel + 2]
			
		rebinneddata = np.zeros((self.numspectra, np.size(rebinning, 0) - 1, np.size(self.detector, 2)), dtype = 'float64')
		rebinneddataSD = np.zeros((self.numspectra, np.size(rebinning, 0) - 1, np.size(self.detector, 2)), dtype = 'float64')			

		#now do the rebinning
		for index in xrange(np.size(self.detector, 0)):
			if self.verbose:
				print datafilenumber, ": rebinning plane: ", index
			#rebin that plane.
			plane, planeSD = rebin.rebin2D(M_lambdaHIST[index], np.arange(np.size(self.detector, 2) + 1.),
			self.detector[index], detectorSD[index], rebinning, np.arange(np.size(self.detector, 2) + 1.))
			assert not np.isnan(planeSD).any()

			rebinneddata[index, ] = plane
			rebinneddataSD[index, ] = planeSD

			self.detector = rebinneddata
			detectorSD = rebinneddataSD

			M_lambdaHIST = np.resize(rebinning, (self.numspectra, np.size(rebinning, 0)))

		#divide the detector intensities by the width of the wavelength bin.
		binwidths = M_lambdaHIST[0, 1:] - M_lambdaHIST[0,:-1]
		self.detector /= binwidths[:,np.newaxis]
		detectorSD /= binwidths[:,np.newaxis]
		
		M_specTOFHIST = qtrans.lambda_to_tof(M_lambdaHIST, chod[:, np.newaxis])
		M_lambda = 0.5 * (M_lambdaHIST[:,1:] + M_lambdaHIST[:,:-1])
		M_spectof = qtrans.lambda_to_tof(M_lambda, chod[:, np.newaxis])
		
		#Now work out where the beam hits the detector
		#this is used to work out the correct angle of incidence.
		#it will be contained in a wave called M_beampos
		#M_beampos varies as a fn of wavelength due to gravity
		
		#TODO work out beam centres for all pixels
		#this has to be done agian because gravity correction is done above.
		if self.isdirect:
		   #the spectral ridge for the direct beam has a gravity correction involved with it.
		   #the correction coefficients for the beamposition are contaned in M_gravcorrcoefs
			M_beampos = np.zeros_like(M_lambda)
			
			# the following correction assumes that the directbeam neutrons are falling from a point position 
			# W_gravcorrcoefs[0] before the detector. At the sample stage (W_gravcorrcoefs[0] - detectorpos[0])
			# they have a certain vertical velocity, assuming that the neutrons had
			# an initial vertical velocity of 0. Although the motion past the sample stage will be parabolic,
			# assume that the neutrons travel in a straight line after that (i.e. the tangent of the parabolic
			# motion at the sample stage). This should give an idea of the direction of the true incident beam,
			# as experienced by the sample.
			#Factor of 2 is out the front to give an estimation of the increase in 2theta of the reflected beam.
			M_beampos[:] = M_gravcorrcoefs[:,1][:, np.newaxis]
			M_beampos[:] -= 2. * (1000. / Y_PIXEL_SPACING * 9.81 * ((M_gravcorrcoefs[:, 0][:, np.newaxis] - detpositions[:, np.newaxis])/1000.) * (detpositions[:, np.newaxis]/1000.) * M_lambda**2/((qtrans.kPlanck_over_MN * 1.e10)**2))
			M_beampos *=  Y_PIXEL_SPACING
		else:
			M_beampos = np.zeros_like(M_lambda)
			M_beampos[:] = beam_centre * Y_PIXEL_SPACING

		#background subtraction
		extent_mult = 2
		if self.background:
			if self.verbose:
				print datafilenumber, ': doing background subtraction'
			self.detector, detectorSD = background_subtract(self.detector, detectorSD, beam_centre, beam_SD, extent_mult, 1)
		
		#top and tail the specular beam with the known beam centres.
		#all this does is produce a specular intensity with shape (n, t), i.e. integrate over specular beam
		lopx = np.floor(beam_centre - beam_SD * extent_mult)
		hipx = np.ceil(beam_centre + beam_SD  * extent_mult)

		M_spec = np.sum(self.detector[:, :, lopx:hipx + 1], axis = 2)
		M_specSD = np.sum(np.power(detectorSD[:, :, lopx:hipx + 1], 2), axis = 2)
		M_specSD = np.sqrt(M_specSD)
		
		assert np.isfinite(M_spec).all()
		assert np.isfinite(M_specSD).all()
		assert np.isfinite(self.detector).all()
		assert np.isfinite(detectorSD).all()

		#
		#normalise by beam monitor 1.
		#
		if self.bmon1_normalise:
			bmon1_countsSD = np.sqrt(self.bmon1_counts)
			#have to make to the same shape as M_spec			
			M_spec, M_specSD = EP.EPdiv(M_spec, M_specSD, self.bmon1_counts[:,np.newaxis], bmon1_countsSD[:,np.newaxis])
			#have to make to the same shape as detector
			#print detector.shape, detectorSD.shape, bmon1_counts[:,np.newaxis, np.newaxis].shape			
			self.detector, detectorSD = EP.EPdiv(self.detector,
											 detectorSD,
											  self.bmon1_counts[:,np.newaxis, np.newaxis],
											   bmon1_countsSD[:,np.newaxis, np.newaxis])

		#now work out dlambda/lambda, the resolution contribution from wavelength.
		#vanWell, Physica B,  357(2005) pp204-207), eqn 4.
		#this is only an approximation for our instrument, as the 2nd and 3rd discs have smaller
		#openings compared to the master chopper.  Therefore the burst time needs to be looked at.
		#W_point should still be the point version of the TOFhistogram.
		M_lambdaSD = ((M_specTOFHIST[:,1:] - M_specTOFHIST[:,:-1]) / M_spectof[:])**2
		#account for the gross resolution of the chopper, adding in a contribution if you have a phase
		#opening.  (don't forget freq is in Hz, W_point is in us.
		#TODO chod might change from scanpoint to scanpoint..... The resolution will be out if you are scanning dy.
		M_lambdaSD += ((D_CX / chod[0]) + (phaseangle / (360 * freq * 1e-6 * M_spectof)))**2
		
		#TODO ss2vg might change from scanpoint to scanpoint..... The resolution will be out if you are scanning ss2vg.
		ss2vg = self.h5data['entry1/instrument/slits/second/vertical/gap']
		tauH = (1e6 * ss2vg[originalscanpoint] / (DISCRADIUS * 2 * np.pi * freq))
		M_lambdaSD += (tauH / M_spectof)**2
		M_lambdaSD *= 0.68**2
		M_lambdaSD = np.sqrt(M_lambdaSD)
		M_lambdaSD *= M_lambda
		
		#put the detector positions and mode into the dictionary as well.
		detectorZ = np.copy(self.h5data['entry1/instrument/detector/vertical_translation'])
		detectorY = np.copy(self.h5data['entry1/instrument/detector/longitudinal_translation'])
		mode = np.copy(self.h5data['entry1/instrument/parameters/mode'])
		
		#if you did event streaming then you will have to expand on the detector positions (they'll all be the same,
		# so it won't matter)
		detectorZ = np.resize(detectorZ, self.numspectra)
		detectorY = np.resize(detectorY, self.numspectra)
		mode = np.resize(mode, self.numspectra)
				
		#create instance variables for information it's useful to have.
		self.M_topandtail = self.detector
		self.M_topandtailSD = detectorSD
		self.M_spec = M_spec
		self.M_specSD = M_specSD
		self.M_beampos = M_beampos
		self.M_lambda = M_lambda
		self.M_lambdaSD = M_lambdaSD
		self.M_lambdaHIST = M_lambdaHIST
		self.M_spectof = M_spectof
		self.mode = mode
		self.detectorZ = detectorZ
		self.detectorY = detectorY
		self.domega = domega
		self.lopx = lopx
		self.hipx = hipx
		self.title = self.h5data['entry1/experiment/title'][0]
		self.sample = self.h5data['entry1/sample/name'][0]
		self.user = self.h5data['entry1/user/name'][0]
		
		self.isprocessed = 1
		self.__nexusClose()
		return self.M_lambda, self.M_lambdaSD, self.M_spec, self.M_specSD
	
	def writeSpectrum(self, f, scanpoint = 0):
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
		if not self.isprocessed:
			return False
		
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
#		filename = 'PLP{:07d}_{:d}.spectrum'.format(self.datafilenumber, index)
			
		d['r'] = string.translate(repr(r[scanpoint].tolist()), None, ',[]')
		d['dr'] = string.translate(repr(dr[scanpoint].tolist()), None, ',[]')
		d['l'] = string.translate(repr(l[scanpoint].tolist()), None, ',[]')
		d['dl'] = string.translate(repr(dl[scanpoint].tolist()), None, ',[]')
		thefile = s.safe_substitute(d)
		f.write(thefile)
		f.truncate()
		
		return True
		

	def processEventStream(self, tbins = None, xbins = None, ybins = None, frame_bins = None, scanpoint = 0):

		had_to_open = False
		if self.h5data.closed:
			self.__nexusOpen()
			had_to_open = True
			
		if not tbins:
			tbins = self.h5data['entry1/data/time_of_flight']
		if not xbins:
			xbins = self.h5data['entry1/data/x_bin']
		if not ybins:
			ybins = self.h5data['entry1/data/y_bin']
		if not frame_bins:
			frame_bins = [0, self.h5data['entry1/instrument/detector/time'][scanpoint]]
		
		total_acquisition_time = self.h5data['entry1/instrument/detector/time'][scanpoint]
		self.frequency = self.h5data['entry1/instrument/disk_chopper/ch1speed'][0] / 60
		bm1counts_for_scanpoint = self.h5data['entry1/monitor/bm1_counts'][scanpoint]

		frame_bins = np.sort(frame_bins)
		
		#truncate the frame bins to be 0 and the max acquisition time, if they exceed it.
		if frame_bins[0] < 0:
			loc = np.searchsorted(frame_bins, 0)
			frame_bins = frame_bins[loc - 1:]
			frame_bins[0] = 0
		
		if frame_bins[-1] > total_acquisition_time:
			loc = np.searchsorted(frame_bins, total_acquisition_time)
			frame_bins = frame_bins[:loc + 1]
			frame_bins[-1] = total_acquisition_time

		bm1_counts = frame_bins[1:] - frame_bins[:-1]
		bm1_counts *= (bm1counts_for_scanpoint / total_acquisition_time)

	
		try:
			eventDirectoryName = self.h5data['entry1/instrument/detector/daq_dirname'][0]
		except KeyError:	#daq_dirname doesn't exist in this file
			if had_to_open:
				self.__nexusClose()
				
			return None
		
		def streamedfileexists(x):
			if os.path.basename(x[0])==eventDirectoryName and 'DATASET_'+str(scanpoint) in x[1]:
				return True
			else:
				return False

		c = filter(streamedfileexists, os.walk(self.basedir))
		if not len(c):
			return None, None, None
		
		streamfilename = os.path.join(c[0][0], 'DATASET_'+str(scanpoint), 'EOS.bin')

		f = open(streamfilename, 'r')
		detector, endoflastevent = self.__nunpack_intodet(f, tbins, ybins, xbins, frame_bins * self.frequency)
		f.close()
		
		if had_to_open:
			self.__nexusClose()

		
		return frame_bins, detector, bm1_counts
		
				
	def __nunpack_intodet(self, f, tbins, ybins, xbins, frame_bins, endoflastevent = 127):	
		if not f:
			return None
					
		run = 1L
		state = 0L
		event_ended = 0L
		frame_number = -1L
		dt = 0L
		t = 0L
		x = -0L
		y = -0L
		
		localxbins = np.array(xbins)
		localybins = np.array(ybins)
		localtbins = np.sort(np.array(tbins))
		localframe_bins = np.array(frame_bins)
			
		if localxbins[0] > localxbins[-1]:
			localxbins = localxbins[::-1]
			reversedX = True

		if localybins[0] > localybins[-1]:
			localybins = localybins[::-1]
			reversedY = True
					
		BUFSIZE=16384
		
		neutrons = []

		while True:	
			if frame_number > localframe_bins[-1]:
				break
			f.seek(endoflastevent + 1)
			buffer = f.read(BUFSIZE)
			
			filepos = endoflastevent + 1
			
			if not len(buffer):
				break
			
			buffer = map(ord, buffer)
			state = 0
			
			for ii, c in enumerate(buffer):
				if state == 0:
					x = c
					state += 1
				elif state == 1:
					x |= (c & 0x3) * 256;
					
					if x & 0x200:
						x = - (0x100000000 - (x | 0xFFFFFC00))
					y = int(c / 4)
					state += 1
				else:
					if state == 2:
						y = y | ((c & 0xF) * 64)

						if y & 0x200:
							y = -(0x100000000 - (y | 0xFFFFFC00))
					event_ended = ((c & 0xC0)!= 0xC0 or state>=7)

					if not event_ended:
						c &= 0x3F
					if state == 2:
						dt = c >> 4
					else:
						dt |= (c) << (2 + 6 * (state - 3));
			
					if not event_ended:
						state += 1;
					else:
						#print "got to state", state, event_ended, x, y, frame_number, t, dt
						state = 0;
						endoflastevent = filepos + ii
						if x == 0 and y == 0 and dt == 0xFFFFFFFF:
							t = 0
							frame_number += 1
						else:
							t += dt;
							if frame_number == -1:
								return None
							neutrons.append((frame_number, t//1000, y, x))
							
		if len(neutrons):
			events = np.array(neutrons)
			histo, edge = np.histogramdd(events, bins=(localframe_bins, localtbins, localybins, localxbins))
		
		if reversedX:
			histo = histo[:,:,:, ::-1]
			
		if reversedY:
			histo = histo[:,:,::-1, :]
		
		return histo, endoflastevent
	
 
	def __nexusOpen(self):
		self.h5data = h5.File(self.datafilename, 'r')
		self.h5data.closed = False		

	def __nexusClose(self):
		self.h5data.close()
		self.h5data.closed = True
		if self.h5norm:
			self.h5norm.close()
			
	def __chodcalculator(self, omega, two_theta, pairing = 10, scanpoint = 0):
		chod = 0
		chopper1_distance = self.h5data['entry1/instrument/parameters/chopper1_distance']
		chopper2_distance = self.h5data['entry1/instrument/parameters/chopper2_distance']
		chopper3_distance = self.h5data['entry1/instrument/parameters/chopper3_distance']
		chopper4_distance = self.h5data['entry1/instrument/parameters/chopper4_distance']
		
		#guide 1 is the single deflection mirror (SB)
		#its distance is from chopper 1 to the middle of the mirror (1m long)
		
		#guide 2 is the double deflection mirror (DB)
		#its distance is from chopper 1 to the middle of the second of the compound mirrors! (a bit weird, I know).
		
		guide1_distance = self.h5data['entry1/instrument/parameters/guide1_distance']
		guide2_distance = self.h5data['entry1/instrument/parameters/guide2_distance']
		sample_distance = self.h5data['entry1/instrument/parameters/sample_distance']
		detectorpos = self.h5data['entry1/instrument/detector/longitudinal_translation']
		mode = self.h5data['entry1/instrument/parameters/mode'][0]#[scanpoint]
		
		#assumes that disk closest to the reactor (out of a given pair) is always master
		for ii in xrange(1,5):
			if pairing & 2**ii:
				master = ii
				break
				
		for ii in xrange(master + 1, 5):
			if pairing & 2**ii:
				slave = ii
				break
		
		if master == 1:
			chod = 0
		elif master == 2:
			chod -= chopper2_distance[0]
		elif master == 3:
			chod -= chopper3_distance[0]
		
			
		if slave == 2:
			chod -= chopper2_distance[0]
		elif slave == 3:
			chod -= chopper3_distance[0]
		elif slave == 4:
			chod -= chopper4_distance[0]
		
				
		#T0 is midway between master and slave, but master may not necessarily be disk 1.
		#However, all instrument lengths are measured from disk1
		chod /= 2
		
		if mode == "FOC" or mode == "POL" or mode == "MT" or mode == "POLANAL":
			chod += sample_distance[0]
			chod += detectorpos[scanpoint] / np.cos(np.pi * two_theta / 180)
		
		elif mode == "SB":   		
			#assumes guide1_distance is in the MIDDLE OF THE MIRROR
			chod += guide1_distance[0]
			chod += (sample_distance[0] - guide1_distance[0]) / np.cos(np.pi * omega / 180)
			if two_theta > omega:
				chod += detectorpos[scanpoint]/np.cos( np.pi* (two_theta - omega) / 180)
			else:
				chod += detectorpos[scanpoint] /np.cos( np.pi * (omega - two_theta) /180)
		
		elif mode == "DB":
			#guide2_distance in in the middle of the 2nd compound mirror
			# guide2_distance - longitudinal length from midpoint1->midpoint2 + direct length from midpoint1->midpoint2
			chod += guide2_distance[0] + 600. * np.cos (1.2 * np.pi/180) * (1 - np.cos(2.4 * np.pi/180)) 
		
			#add on distance from midpoint2 to sample
			chod +=  (sample_distance[0] - guide2_distance[0]) / np.cos(4.8 * np.pi/180)
			
			#add on sample -> detector			
			if two_theta > omega:			
				chod += detectorpos[scanpoint] / np.cos( np.pi* (two_theta - 4.8) / 180)
			else:
				chod += detectorpos[scanpoint] / np.cos( np.pi * (4.8 - two_theta) /180)
		
		return chod

	
	def __createdetectornorm(self, xmin, xmax):
		"""
		produces a detector normalisation for Platypus
		you give it a water run and it average n, t and x to provide 
		a relative efficiency for each y wire.
		"""		
		#average over n and TOF and x
		#n
		norm = np.sum(np.array(h5norm['entry1/data/hmm']), axis = 0, dtype = 'float64')
		#t
		norm = np.sum(norm, axis = 0, dtype = 'float64')
		# by this point you have norm[y][x]
		norm = norm[:, xmin:xmax + 1]
		norm = np.sum(norm, axis = 1, dtype = 'float64')    
			
		normSD = np.empty_like(norm)
		normSD = np.sqrt(norm)
		
		norm /= mean
		normSD /= mean		
		return norm, normSD


def background_subtract(detector, detectorSD, beam_centre, beam_SD, extent_mult = 2., pixel_offset = 1.):
	"""
	shape of detector is (n, t, y)
	does a linear background subn for each (n, t) slice
	"""
	ret_array = np.zeros(detector.shape, dtype = 'float64')
	retSD_array = np.zeros(detector.shape, dtype = 'float64')
	
	for index in np.ndindex(detector.shape[0:2]):
		yslice = detector[index]
		ySDslice = detectorSD[index]
		ret_array[index], retSD_array[index] = background_subtract_line(yslice, ySDslice, beam_centre, beam_SD, extent_mult, pixel_offset)
				
	return ret_array, retSD_array
	
def background_subtract_line(detector, detectorSD, beam_centre, beam_SD, extent_mult = 2., pixel_offset = 1.):

	lopx = np.floor(beam_centre - beam_SD * extent_mult)
	hipx = np.ceil(beam_centre + beam_SD  * extent_mult)
	
	y0 = round(lopx - (extent_mult * extent_mult * beam_SD) - pixel_offset - 1)
	y1 = round(lopx - pixel_offset - 1)
	y2 = round(hipx + pixel_offset + 1)
	y3 = round(hipx + (extent_mult * extent_mult * beam_SD) + pixel_offset + 1)
	
	xvals = np.array([x for x in xrange(len(detector)) if (y0 <= x < y1 or y2 < x <= y3)], dtype = 'int')
	yvals = detector[xvals]
	ySDvals = detectorSD[xvals] 
	xvals = np.asfarray(xvals)
	
	#some SD values may have 0 SD, which will screw up curvefitting.
	ySDvals = np.where(ySDvals == 0, 1, ySDvals)
	assert not np.isnan(ySDvals).any()
		
	#equation for a straight line
	f = lambda x, a, b: a + b * x
	
	#estimate the linear fit
	y_bar = np.mean(yvals)
	x_bar = np.mean(xvals)
	bhat = np.sum((xvals - x_bar) * (yvals - y_bar)) / np.sum((xvals - x_bar)**2)
	ahat = y_bar - bhat * x_bar
	
	#get the weighted fit values
	popt, pcov = curve_fit(f, xvals, yvals, sigma = ySDvals, p0 = np.array([ahat, bhat]))
		
	#SD of params = np.sqrt(chi2) * np.sqrt(pcov)
	#chi2 = lambda ycalc, yobs, sobs: np.sum(((ycalc - yobs)/sobs)**2)
	CI = lambda x, pcovmat: (np.matrix([1., x]) * np.asmatrix(pcovmat) * np.matrix([1., x]).T)[0,0]
	
	bkgd = f(np.arange(len(detector), dtype = 'float64'), popt[0], popt[1])
	bkgdSD = np.empty_like(bkgd)
	
	#if you try to do a fit which has a singular matrix
	if np.isfinite(pcov).all():
		bkgdSD = np.asarray([CI(x, pcov) for x in np.arange(len(detector))], dtype = 'float64')
	else:
		bkgdSD = np.zeros_like(bkgd)

	bkgdSD = np.sqrt(bkgdSD)
	#get the t value for a two sided student t test at the 68.3 confidence level
	
	bkgdSD *= t.isf(0.1585, len(xvals) - 2)
	
	return EP.EPsub(detector, detectorSD, bkgd, bkgdSD)

def findspecularridge(detector, startingoffset = None, tolerance = 0.01):
	"""
	find the specular ridge in a detector(n, t, y) plot.
	"""
	
	searchincrement = 50
	#sum over all n planes, left with ty
	det_ty = np.sum(detector, axis = 0)
	
	if not startingoffset:
		startingoffset = 50
	else:
		startingoffset = abs(startingoffset)
		
	numincrements = (len(det_ty) - startingoffset) // searchincrement
	
	for ii in xrange(numincrements):
		totaly = np.sum(det_ty[-1: -startingoffset - searchincrement * ii: -1], axis = 0)
		#find the centroid and gauss peak in the last sections of the TOF plot
		centroid, gausspeak = ut.peakfinder(totaly)
			
		if ii and abs((gausspeak[0] - lastcentre) / lastcentre) < tolerance and abs((gausspeak[1] - lastSD) / lastSD) < tolerance:
			lastcentre = gausspeak[0]
			lastSD = gausspeak[1]
			break
		
		lastcentre = gausspeak[0]
		lastSD = gausspeak[1]
	
	
	return lastcentre, lastSD
	
def correct_for_gravity(detector, detectorSD, lamda, trajectory, lolambda, hilambda):
	'''
	this function provides a gravity corrected yt plot, given the data, its associated errors, the wavelength corresponding to each of the time bins, and the trajectory of the neutrons.  Low lambda and high Lambda are wavelength cutoffs to igore.
	
	output:
	corrected data, dataSD
	M_gravCorrCoefs.  THis is a theoretical prediction where the spectral ridge is for each timebin.  This will be used to calculate the actual angle of incidence in the reduction process.
	
	data has shape (n, t, y)
	M_lambda has shape (n, t)
	'''
	numlambda = np.size(lamda, axis = 1)
	
	x_init = np.arange((np.size(detector, axis = 2) + 1) * 1.) - 0.5
	
	f = lambda x, td, tru_centre: deflection(x, td, 0) / Y_PIXEL_SPACING + tru_centre
	
	M_gravcorrcoefs = np.zeros((len(detector), 2), dtype = 'float64')
	
	correcteddata = np.empty_like(detector)
	correcteddataSD = np.empty_like(detectorSD) 

	for spec in xrange(len(detector)):
		#centres(t,)
		centroids = np.apply_along_axis(ut.centroid, 1, detector[spec])
		lopx = np.trunc(np.interp(lolambda, lamda[spec], np.arange(numlambda)))
		hipx = np.ceil(np.interp(hilambda, lamda[spec], np.arange(numlambda)))
		
		M_gravcorrcoefs[spec], pcov = curve_fit(f, lamda[spec,lopx:hipx], centroids[:, 0][lopx:hipx], np.array([3000., np.mean(centroids)]))
		totaldeflection = deflection(lamda[spec], M_gravcorrcoefs[spec][0], 0) / Y_PIXEL_SPACING

		for wavelength in xrange(np.size(detector, axis = 1)):
			x_rebin = x_init + totaldeflection[wavelength]
			correcteddata[spec,wavelength], correcteddataSD[spec,wavelength] = rebin.rebin(x_init, detector[spec,wavelength], detectorSD[spec, wavelength], x_rebin)
	
	return correcteddata, correcteddataSD, M_gravcorrcoefs
	
def deflection(lamda, travel_distance, trajectory):
	#returns the deflection in mm of a ballistic neutron
	#lambda in Angstrom, travel_distance (length of correction, e.g. sample - detector) in mm, trajectory in degrees above the horizontal
	#The deflection correction  is the distance from where you expect the neutron to hit the detector (detector_distance*tan(trajectory)) to where is actually hits the detector, i.e. the vertical deflection of the neutron due to gravity.

	trajRad = trajectory * np.pi/180
	pp = travel_distance/1000. * np.tan(trajRad)
	
	pp -= 9.81* (travel_distance/1000.)**2 * (lamda/1.e10)**2 / (2*np.cos(trajRad)*np.cos(trajRad)*(qtrans.kPlanck_over_MN)**2)
	pp *= 1000

	return pp
	

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some Platypus NeXUS files to produce their TOF spectra.')
    parser.add_argument('file_list', metavar='N', type=int, nargs='+',
                   help='integer file numbers')
    parser.add_argument('--basedir', type=str, help='define the location to find the nexus files')
    parser.add_argument('--rebin', type=float, help='rebin percentage for the wavelength -1<rebin<10', default = 4)
    parser.add_argument('--lolambda', type=float, help='lo wavelength cutoff for the rebinning', default=2.8)
    parser.add_argument('--hilambda', type=float, help='lo wavelength cutoff for the rebinning', default=18.)
    parser.add_argument('--typeofintegration', type=float, help='0 to integrate all spectra, 1 to output individual spectra', default=0)
    args = parser.parse_args()
    print args
    
    for file in args.file_list:
        print 'processing: %d' % file
        try:
			a = processnexus(file,           
							   basedir = args.basedir)
		
			M_lambda, M_lambdaSD, M_spec, M_specSD = a.process(lolambda = args.lolambda,
					   hilambda = args.hilambda,
						rebinpercent = args.rebin,
						 typeofintegration = args.typeofintegration)
			
			for index in xrange(a.numspectra):
				filename = 'PLP{:07d}_{:d}.spectrum'.format(a.datafilenumber, index)
				f = open(filename, 'w')
				a.writeSpectrum(f, scanpoint = index)
				f.close()
							
        except IOError:
            print 'Couldn\'t find file: %d.  Use --basedir option' %file
        

