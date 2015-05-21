from __future__ import division
import numpy as np
import h5py as h5
import pyplatypus.util.ErrorProp as EP
import peak_utils as ut
import processnexus
import platypusspectrum
import Qtransforms as qtrans
from scipy.optimize import curve_fit
from scipy.stats import t
import rebin
import string
from time import gmtime, strftime
import os
import os.path
import argparse
import re

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

class ProcessPlatypusNexus(processnexus.ProcessNexus):

	"""
	
		This class is a processor for a Platypus nexus file to produce an intensity vs wavelength spectrum. 
		
		Usage:
		>>>h5data = h5py.File('PLP0000708.nx.hdf','r')
		>>>processorObject = ProcessPlatypusNexus()
		>>>spectrum = processorObject.process(h5data)
		>>>spectrum.write_spectrum()

		
	"""
		
	def __init__(self):
		super(ProcessPlatypusNexus, self).__init__
	
	def catalogue(self, h5data, scanpoint = 0):
		d = {}

		try:
			d['filename'] = h5data.filename

			path, datafilenumber = os.path.split(d['filename'])
			reg = re.compile('PLP(\d+).nx.hdf')
			d['datafilenumber'] = int(re.split(reg, datafilenumber)[1])
			d['end_time'] = h5data['entry1/end_time'][0]
			d['ss1vg'] = h5data['entry1/instrument/slits/first/vertical/gap'][scanpoint]
			d['ss2vg'] = h5data['entry1/instrument/slits/second/vertical/gap'][scanpoint]
			d['ss3vg'] = h5data['entry1/instrument/slits/third/vertical/gap'][scanpoint]
			d['ss4vg'] = h5data['entry1/instrument/slits/fourth/vertical/gap'][scanpoint]		
			d['ss1hg'] = h5data['entry1/instrument/slits/first/horizontal/gap'][scanpoint]
			d['ss2hg'] = h5data['entry1/instrument/slits/second/horizontal/gap'][scanpoint]
			d['ss3hg'] = h5data['entry1/instrument/slits/third/horizontal/gap'][scanpoint]
			d['ss4hg'] = h5data['entry1/instrument/slits/fourth/horizontal/gap'][scanpoint]		

			d['sth'] = h5data['entry1/sample/sth'][scanpoint]
			d['bm1_counts'] = h5data['entry1/monitor/bm1_counts'][scanpoint]
			d['total_counts'] = h5data['entry1/instrument/detector/total_counts'][scanpoint]
			d['time'] = h5data['entry1/instrument/detector/time'][scanpoint]
			d['mode'] = h5data['entry1/instrument/parameters/mode'][0]
			d['daq_dirname'] = h5data['entry1/instrument/detector/daq_dirname'][0]
		except:
			pass
			
		return d
	
	def process(self, h5data, h5norm = None, **kwds):
		"""
		
			Processes the ProcessNexus object to produce a time of flight spectrum.
			This method returns an instance of PlatypusSpectrum.
			
			h5data - the nexus file for the data.
			
			normfile - the file containing the floodfield data.
				
			THe following keywords can be set to influence the processing:
				basedir						- base directory where files are found.
							
				lolambda					-the low wavelength cutoff for the rebinned data (A), default = 2.8

				hilambda					-the high wavelength cutoff for the rebinned data (A), default = 18.0

				background					-should a background subtraction be carried out, default = True

				isdirect					-is it a direct beam you measured. This is so a gravity correction can be applied, default = False

				eventstreaming				-do you want to use the listmode data file to create a detector image from a subperiod within
											 the acquisition. This is useful for kinetic experiments. The listmode datafile must
											 reside in a subdirectory of self.basedir.
											 The eventstreaming keyword is specified as a dictionary:
											
											eventstreaming = {'scanpoint':0, 'frame_bins':[5, 10, 120]}
											
											The scanpoint value indicates which scanpoint in the nexus file you want to analyse.
											The frame_bins list specifies the bin edges (in seconds) for the periods you wish to generate
											detector images for. The above example will create two detector images (N=2), the first starts
											at 5s and finishes at 10s. The second starts at 10s and finishes at 120s. The frame_bins are
											clipped to the total acquisition time if necessary. default = False

				peak_pos(pos, sd)			-tuple specifying the location of the specular beam and it's standard deviation. If this is not
											specified (recommended) then an auto specular beam finder is used. default = None
											
				omega						-expected angle of incidence of the beam, default = 0
				
				twotheta					-expected twotheta value of the beam, default = 0
				
				rebinpercent				-rebins the data in constant dlambda/lambda, default = 4%
				
				wavelengthbins				-list containing bin edges for wavelength rebinning. If these are specified then the rebinning is directed
											from these wavelength bins. This option trumps rebin percent. default = None
											
				bmon1_normalise				-normalise by beam monitor 1. Default = True
				
				verbose						-print out information during processing. Default = False
								
				typeofintegration			-if type of integration = 0, then the output spectrum is summed over all scanpoints (N=1)
											 if type of integration = 1, then the spectra from each scanpoint are not summed (N >= 1)
											 eventstreaming trumps this option. Default = 0.
				
			
		
		"""
		self.basedir = kwds.get('basedir', os.getcwd())
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

		frequency = h5data['entry1/instrument/disk_chopper/ch1speed'][0]
		
		scanpoint = 0
		
		#beam monitor counts for normalising data
		bmon1_counts = np.zeros(dtype = 'float64', shape = h5data['entry1/monitor/bm1_counts'].shape)
		bmon1_counts = h5data['entry1/monitor/bm1_counts'][:]
		
		#set up the RAW TOF bins (n, t + 1)
		TOF = np.zeros(dtype = 'float64', shape = h5data['entry1/data/time_of_flight'].shape)
		TOF = h5data['entry1/data/time_of_flight'][:]

		#event streaming.
		if 'eventstreaming' in kwds:
			scanpoint = self.eventstreaming['scanpoint']
			frame_bins, detector, bmon1_counts = self.process_event_stream(h5data, scanpoint = self.eventstreaming['scanpoint'], frame_bins = self.eventstreaming['frame_bins'])
			bmon1_counts = np.array((bmon1_counts), dtype = 'float64')

			numspectra = len(detector)
		else:
			#detector(n, t, y, x)    
			detector = np.zeros(dtype='int32', shape = h5data['entry1/data/hmm'].shape)
			detector = h5data['entry1/data/hmm'][:]
			
			#you average over the individual measurements in the file
			if self.typeofintegration == 0:
				numspectra = 1
				detector = np.sum(detector, axis = 0)
				detector = np.resize(detector, (1, np.size(detector, 0), np.size(detector, 1), np.size(detector, 2)))
				bmon1_counts = np.array([np.sum(bmon1_counts)], dtype = 'float64')
			else:
				numspectra = len(detector)

		#pre-average over x, leaving (n, t, y) also convert to dp
		detector = np.sum(detector, axis = 3, dtype = 'float64')

		#detector shape should now be (n, t, y)
		#create the SD of the array
		detectorSD = np.sqrt(detector + 1)
		
		#detector normalisation with a water file
		if h5norm:
			xbins = h5data['entry1/data/x_bin']
			#shape (y,)
			M_detectornorm, M_detectornormSD = createdetectornorm(h5norm, xbins[0], xbins[1])
			#detector has shape (n,t,y), shape of M_waternorm should broadcast to (1,1,y)
			detector, detectorSD = EP.EPdiv(detector, detectorSD, M_detectornorm, M_detectornormSD)
					
		#shape of these is (numspectra, TOFbins)
		M_specTOFHIST = np.zeros((numspectra, len(TOF)), dtype = 'float64')
		M_lambdaHIST = np.zeros((numspectra, len(TOF)), dtype = 'float64')
		M_specTOFHIST[:] = TOF

		#chopper to detector distances
		#note that if eventstreaming is specified the numspectra is NOT
		#equal to the number of entries in e.g. /longitudinal_translation
		#this means you have to copy values in from the same scanpoint
		chod = np.zeros(numspectra, dtype = 'float64')
		detpositions = np.zeros(numspectra, dtype = 'float64')
		
		#domega, the angular divergence of the instrument
		domega = np.zeros(numspectra, dtype = 'float64')
		
		#process each of the spectra taken in the detector image
		originalscanpoint = scanpoint
		for index in xrange(numspectra):
			if self.verbose:
				print datafilenumber, ': processing image for tof params: ', index
			omega = h5data['entry1/instrument/parameters/omega'][scanpoint]
			two_theta = h5data['entry1/instrument/parameters/twotheta'][scanpoint]
			frequency = h5data['entry1/instrument/disk_chopper/ch1speed']
			ch2speed = h5data['entry1/instrument/disk_chopper/ch2speed']
			ch3speed = h5data['entry1/instrument/disk_chopper/ch3speed']
			ch4speed = h5data['entry1/instrument/disk_chopper/ch4speed']
			ch2phase = h5data['entry1/instrument/disk_chopper/ch2phase']
			ch3phase = h5data['entry1/instrument/disk_chopper/ch3phase']
			ch4phase = h5data['entry1/instrument/disk_chopper/ch4phase']
			ch1phaseoffset = h5data['entry1/instrument/parameters/chopper1_phase_offset']
			ch2phaseoffset = h5data['entry1/instrument/parameters/chopper2_phase_offset']
			ch3phaseoffset = h5data['entry1/instrument/parameters/chopper3_phase_offset']
			ch4phaseoffset = h5data['entry1/instrument/parameters/chopper4_phase_offset']
			chopper1_distance = h5data['entry1/instrument/parameters/chopper1_distance']
			chopper2_distance = h5data['entry1/instrument/parameters/chopper2_distance']
			chopper3_distance = h5data['entry1/instrument/parameters/chopper3_distance']
			chopper4_distance = h5data['entry1/instrument/parameters/chopper4_distance']        
			ss2vg = h5data['entry1/instrument/slits/second/vertical/gap']
			ss3vg = h5data['entry1/instrument/slits/third/vertical/gap']
			slit2_distance = h5data['/entry1/instrument/parameters/slit2_distance']
			slit3_distance = h5data['/entry1/instrument/parameters/slit3_distance']
			detectorpos = h5data['entry1/instrument/detector/longitudinal_translation']

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
			if 'entry1/instrument/parameters/slave' in h5data and 'entry1/instrument/parameters/master' in h5data:
				master = h5data['entry1/instrument/parameters/master'][scanpoint]
				slave = h5data['entry1/instrument/parameters/slave'][scanpoint]
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
			chod[index] = chodcalculator(h5data, omega, two_theta, pairing, scanpoint)
		
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

#		assert not np.isnan(detectorSD).any()
#		assert not np.less(detectorSD, 0).any()
	
		#get the specular ridge on the averaged detector image
		if self.peak_pos:
			beam_centre, beam_SD = peak_pos
		else:
			startingoffset = np.searchsorted(M_lambdaHIST[0], self.hilambda)
			beam_centre, beam_SD = find_specular_ridge(detector, 500)
			if self.verbose:
				print datafilenumber, ": BEAM_CENTRE", datafilenumber, beam_centre
		
		#TODO gravity correction if direct beam
		if self.isdirect:
			detector, detectorSD, M_gravcorrcoefs = correct_for_gravity(detector, detectorSD, M_lambda, 0, 2.8, 18)
			beam_centre, beam_SD = find_specular_ridge(detector)
			
		#rebinning in lambda for all detector
		#rebinning is the default option, but sometimes you don't want to.
		#detector shape input is (n, t, y)
		#we want to rebin t.
		if 'wavelengthbins' in kwds:
			rebinning = kwds['wavelengthbins']
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
			
		rebinneddata = np.zeros((numspectra, np.size(rebinning, 0) - 1, np.size(detector, 2)), dtype = 'float64')
		rebinneddataSD = np.zeros((numspectra, np.size(rebinning, 0) - 1, np.size(detector, 2)), dtype = 'float64')			

		#now do the rebinning
		for index in xrange(np.size(detector, 0)):
			if self.verbose:
				print datafilenumber, ": rebinning plane: ", index
			#rebin that plane.
			plane, planeSD = rebin.rebin2D(M_lambdaHIST[index], np.arange(np.size(detector, 2) + 1.),
			detector[index], detectorSD[index], x_rebin = rebinning)
#			assert not np.isnan(planeSD).any()

			rebinneddata[index, ] = plane
			rebinneddataSD[index, ] = planeSD

		detector = rebinneddata
		detectorSD = rebinneddataSD

		M_lambdaHIST = np.resize(rebinning, (numspectra, np.size(rebinning, 0)))

		#divide the detector intensities by the width of the wavelength bin.
		binwidths = M_lambdaHIST[0, 1:] - M_lambdaHIST[0,:-1]
		detector /= binwidths[:,np.newaxis]
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
			detector, detectorSD = background_subtract(detector, detectorSD, beam_centre, beam_SD, extent_mult, 1)
		
		#top and tail the specular beam with the known beam centres.
		#all this does is produce a specular intensity with shape (n, t), i.e. integrate over specular beam
		lopx = np.floor(beam_centre - beam_SD * extent_mult)
		hipx = np.ceil(beam_centre + beam_SD  * extent_mult)

		M_spec = np.sum(detector[:, :, lopx:hipx + 1], axis = 2)
		M_specSD = np.sum(np.power(detectorSD[:, :, lopx:hipx + 1], 2), axis = 2)
		M_specSD = np.sqrt(M_specSD)
		
#		assert np.isfinite(M_spec).all()
#		assert np.isfinite(M_specSD).all()
#		assert np.isfinite(detector).all()
#		assert np.isfinite(detectorSD).all()

		#
		#normalise by beam monitor 1.
		#
		if self.bmon1_normalise:
			bmon1_countsSD = np.sqrt(bmon1_counts)
			#have to make to the same shape as M_spec			
			M_spec, M_specSD = EP.EPdiv(M_spec, M_specSD, bmon1_counts[:,np.newaxis], bmon1_countsSD[:,np.newaxis])
			#have to make to the same shape as detector
			#print detector.shape, detectorSD.shape, bmon1_counts[:,np.newaxis, np.newaxis].shape			
			detector, detectorSD = EP.EPdiv(detector,
											 detectorSD,
											  bmon1_counts[:,np.newaxis, np.newaxis],
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
		ss2vg = h5data['entry1/instrument/slits/second/vertical/gap']
		tauH = (1e6 * ss2vg[originalscanpoint] / (DISCRADIUS * 2 * np.pi * freq))
		M_lambdaSD += (tauH / M_spectof)**2
		M_lambdaSD *= 0.68**2
		M_lambdaSD = np.sqrt(M_lambdaSD)
		M_lambdaSD *= M_lambda
		
		#put the detector positions and mode into the dictionary as well.
		detectorZ = np.copy(h5data['entry1/instrument/detector/vertical_translation'])
		detectorY = np.copy(h5data['entry1/instrument/detector/longitudinal_translation'])
		mode = np.copy(h5data['entry1/instrument/parameters/mode'])
		
		#if you did event streaming then you will have to expand on the detector positions (they'll all be the same,
		# so it won't matter)
		detectorZ = np.resize(detectorZ, numspectra)
		detectorY = np.resize(detectorY, numspectra)
		mode = np.resize(mode, numspectra)
				
		#create instance variables for information it's useful to have.
		filename = h5data['/entry1/experiment/file_name'][0]
		path, datafilenumber = os.path.split(filename)
		reg = re.compile('PLP(\d+).nx.hdf')
		datafilenumber = int(re.split(reg, datafilenumber)[1])
		
		d = dict()
		d['datafilename'] = h5data.filename
		d['datafilenumber'] = datafilenumber
		
		if h5norm:
			d['normfilename'] = h5norm.filename
		d['M_topandtail'] = detector
		d['M_topandtailSD'] = detectorSD
		d['numspectra'] = numspectra
		d['bmon1_counts'] = bmon1_counts
		d['M_spec'] = M_spec
		d['M_specSD'] = M_specSD
		d['M_beampos'] = M_beampos
		d['M_lambda'] = M_lambda
		d['M_lambdaSD'] = M_lambdaSD
		d['M_lambdaHIST'] = M_lambdaHIST
		d['M_spectof'] = M_spectof
		d['mode'] = mode
		d['detectorZ'] = detectorZ
		d['detectorY'] = detectorY
		d['domega'] = domega
		d['lopx'] = lopx
		d['hipx'] = hipx
		d['title'] = h5data['entry1/experiment/title'][0]
		d['sample'] = h5data['entry1/sample/name'][0]
		d['user'] = h5data['entry1/user/name'][0]
		
		return platypusspectrum.PlatypusSpectrum(**d)
		

	def process_event_stream(self, h5data, tbins = None, xbins = None, ybins = None, frame_bins = None, scanpoint = 0):
		"""
			
			Processes the listmode dataset for the nexus file. It creates a new detector image based on the
			tbins, xbins, ybins and frame_bins you supply to the method (these should all be lists/numpy arrays
			specifying the edges of the required bins).
				If these are not specified, then the default bins are taken from the nexus file (assumed to reside in a subdirectory
			below self.basedir). This would essentially return the same detector image as the nexus file.
				However, you can specify the frame_bins list to generate detector images based on subdivided periods of 
			the total acquisition. For example if framebins = [5, 10, 120] you will get 2 images.  The first starts
			at 5s and finishes at 10s. The second starts at 10s and finishes at 120s. The frame_bins are
			clipped to the total acquisition time if necessary.
			
			Returns:
				frame_bins[N], detector[N, T, Y, X], bm1_counts[N]

			where N is the number of frame_bins, T is the number of time-of-flight bins, Y is the number of y pixel bins,
			X is the number of x pixel bins. bm1_counts is the divided bm1 counts for use in normalisation.
			
		"""
		
		if not tbins:
			tbins = h5data['entry1/data/time_of_flight']
		if not xbins:
			xbins = h5data['entry1/data/x_bin']
		if not ybins:
			ybins = h5data['entry1/data/y_bin']
		if not frame_bins:
			frame_bins = [0, h5data['entry1/instrument/detector/time'][scanpoint]]
		
		total_acquisition_time = h5data['entry1/instrument/detector/time'][scanpoint]
		frequency = h5data['entry1/instrument/disk_chopper/ch1speed'][0] / 60
		bm1counts_for_scanpoint = h5data['entry1/monitor/bm1_counts'][scanpoint]
	
		try:
			eventDirectoryName = h5data['entry1/instrument/detector/daq_dirname'][0]
		except KeyError:	#daq_dirname doesn't exist in this file					
			return None
				
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

		
		def streamedfileexists(x):
			if os.path.basename(x[0])==eventDirectoryName and 'DATASET_'+str(scanpoint) in x[1]:
				return True
			else:
				return False

		c = filter(streamedfileexists, os.walk(self.basedir))
		if not len(c):
			return None, None, None
		

		streamfilename = os.path.join(c[0][0], 'DATASET_'+str(scanpoint), 'EOS.bin')
		detector = None		
		with open(streamfilename, 'r') as f: 
			detector, endoflastevent = self.__nunpack_intodet(f, tbins, ybins, xbins, frame_bins * frequency)
		
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


def createdetectornorm(h5norm, xmin, xmax):
	"""
	produces a detector normalisation for Platypus
	you give it a water run and it average n, t and x to provide 
	a relative efficiency for each y wire.
	"""

	#average over n and TOF and x
	#n
	norm = np.sum(np.array(h5norm['entry1/data/hmm']), axis=0,
                  dtype='float64')
	#t
	norm = np.sum(norm, axis=0, dtype='float64')

	# by this point you have norm[y][x]
	norm = norm[:, xmin:xmax + 1]
	norm = np.sum(norm, axis=1, dtype='float64')
		
	normSD = np.sqrt(norm)

	mean = np.mean(norm)
	norm /= mean
	normSD /= mean		
		
	return norm, normSD
		
			
def chodcalculator(h5data, omega, two_theta, pairing = 10, scanpoint = 0):
	chod = 0
	chopper1_distance = h5data['entry1/instrument/parameters/chopper1_distance']
	chopper2_distance = h5data['entry1/instrument/parameters/chopper2_distance']
	chopper3_distance = h5data['entry1/instrument/parameters/chopper3_distance']
	chopper4_distance = h5data['entry1/instrument/parameters/chopper4_distance']
	
	#guide 1 is the single deflection mirror (SB)
	#its distance is from chopper 1 to the middle of the mirror (1m long)
	
	#guide 2 is the double deflection mirror (DB)
	#its distance is from chopper 1 to the middle of the second of the compound mirrors! (a bit weird, I know).
	
	guide1_distance = h5data['entry1/instrument/parameters/guide1_distance']
	guide2_distance = h5data['entry1/instrument/parameters/guide2_distance']
	sample_distance = h5data['entry1/instrument/parameters/sample_distance']
	detectorpos = h5data['entry1/instrument/detector/longitudinal_translation']
	mode = h5data['entry1/instrument/parameters/mode'][0]#[scanpoint]
	
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
#	assert not np.isnan(ySDvals).any()
		
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
#	CI = lambda x, pcovmat: (np.matrix([1., x]) * np.asmatrix(pcovmat) * np.matrix([1., x]).T)[0,0]
	CI = lambda x, pcovmat: pcovmat[0, 0] + pcovmat[1,0] * x + pcovmat[0, 1] * x + pcovmat[1, 1] * (x**2)
	
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

def find_specular_ridge(detector, startingoffset = None, tolerance = 0.01):
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
		centroid, gausspeak = ut.peak_finder(totaly)
			
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

	
def catalogue_all(basedir = None, fname = None):
	
	if not basedir:
		files_to_catalogue = [filename for filename in os.listdir(os.getcwd()) if is_platypus_file(filename)]
	else:
		files_to_catalogue = []
		for root, dirs, files in os.walk(basedir):
			files_to_catalogue.append([os.path.join(root, filename) for filename in files if is_platypus_file(filename)])
		
	files_to_catalogue = [item for sublist in files_to_catalogue for item in sublist]
	filenumbers = [is_platypus_file(filename) for filename in files_to_catalogue]
	
	Tppn = ProcessPlatypusNexus()
	
	listdata = []
	
	for filename in files_to_catalogue:
		try:
			with h5.File(filename, 'r') as h5data:
				listdata.append((is_platypus_file(filename), Tppn.catalogue(h5data)))
				h5data.close()
		except:
			pass
			
	uniquelist = []
	uniquefnums = []
	for item in listdata:
		if not item[0] in uniquefnums:
			uniquelist.append(item)
			uniquefnums.append(item[0])
	
	uniquelist.sort()
	if fname:
		template = """$datafilenumber\t$end_time\t$ss1vg\t$ss2vg\t$ss3vg\t$ss4vg\t$total_counts\t$bm1_counts\t$time\t$mode\t$daq_dirname\n"""
		with open(fname, 'w') as f:
			f.write(template)
			s = string.Template(template)

			for item in uniquelist:
				f.write(s.safe_substitute(item[1]))
			
			f.truncate()

	return uniquelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some Platypus NeXUS files to produce their TOF spectra.')
    parser.add_argument('file_list', metavar='N', type=int, nargs='+',
                   help='integer file numbers')
    parser.add_argument('--basedir', type=str, help='define the location to find the nexus files')
    parser.add_argument('--rebinpercent', type=float, help='rebin percentage for the wavelength -1<rebin<10', default = 4)
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
				a.writespectrum(f, scanpoint = index)
				f.close()
							
        except IOError:
            print 'Couldn\'t find file: %d.  Use --basedir option' %file
        

