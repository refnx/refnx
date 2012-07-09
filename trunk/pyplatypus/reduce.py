from __future__ import division
import numpy as np
import h5py
import processplatypusnexus as ppn
import platypusspectrum as ps
import ErrorProp as EP
import Qtransforms as qtrans
import string
from time import gmtime, strftime
import reflectdataset as rd
import os
    
class Reduce(object):
	def __init__(self, h5ref, h5direct, h5norm = None, **kwds):
		keywords = kwds.copy()
		keywords['isdirect'] = False
		
		#create a processing object
		processingObj = ppn.processplatypusnexus(**keywords)
		
		#get the spectrum for the reflected beam
		self.reflect_beam = processingObj.process(h5ref, h5norm = h5norm, **keywords)
		
		#now get the spectrum for the direct beam
		keywords['isdirect'] = True
		keywords['wavelengthbins'] = self.reflect_beam.M_lambdaHIST
		del(keywords['eventstreaming'])
	
		self.direct_beam = processingObj.process(h5direct, h5norm = h5norm, **kwds)
		
		self.__reduce_single_angle()
		
		
	def get_1D_data(self, scanpoint = 0):
		return (self.W_q[scanpoint], self.W_ref[scanpoint], self.W_refSD[scanpoint], self.W_qSD[scanpoint])

	def get_2D_data(self, scanpoint = 0):
		return (self.M_qz[scanpoint], self.M_qy[scanpoint], self.M_ref[scanpoint], self.M_refSD[scanpoint])
		
	def scale(self, scale):
		self.M_ref /= scale
		self.M_refSD /=scale
		self.W_ref /=scale
		self.W_refSD /= scale
		
	def get_reflected_dataset(self, scanpoint = 0):
		reflectedDatasetObj = rd.ReflectDataset()
		reflectedDatasetObj.add_dataset(self, scanpoint = scanpoint)
		return reflectedDatasetObj
	
	def write_offspecular(self, f, scanpoint = 0):
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
		<Qy type="FWHM" units="1/A">$_qy</Qy>
		</REFdata>
		</REFentry>
		</REFroot>"""
		d = dict()
		d['time'] = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
		d['_rnumber'] = self.reflect_beam.datafilenumber
		d['_numpointsz'] = np.size(self.M_ref, 1)
		d['_numpointsy'] = np.size(self.M_ref, 2) 
		
		s = string.Template(__template_ref_xml)

#			filename = 'off_PLP{:07d}_{:d}.xml'.format(self._rnumber, index)
		d['_r'] = string.translate(repr(self.M_ref[scanpoint].tolist()), None, ',[]')
		d['_qz'] = string.translate(repr(self.M_qz[scanpoint].tolist()), None, ',[]')
		d['_dr'] = string.translate(repr(self.M_refSD[scanpoint].tolist()), None, ',[]')
		d['_qy'] = string.translate(repr(self.M_qy[scanpoint].tolist()), None, ',[]')
		
		thefile = s.safe_substitute(d)
		f.write(thefile)
		f.truncate()
			
	def __reduce_single_angle(self):
		numspectra = self.reflect_beam.numspectra
		numtpixels = np.size(self.reflect_beam.M_topandtail, 1)
		numypixels = np.size(self.reflect_beam.M_topandtail, 2)
		
		#calculate omega and two_theta depending on the mode.
		mode = self.reflect_beam.mode
		M_twotheta = np.zeros(self.reflect_beam.M_topandtail.shape, dtype = 'float64')
		
		if mode == 'FOC' or mode == 'POL' or mode == 'POLANAL' or mode == 'MT':
			omega = self.reflect_beam.M_beampos + self.reflect_beam.detectorZ[:, np.newaxis]
			omega -= self.direct_beam.M_beampos + self.direct_beam.detectorZ
			omega /= self.reflect_beam.detectorY[:, np.newaxis]

			omega = np.arctan(omega) / 2
			#print reflect_beam['M_beampos'], reflect_beam['detectorZ']
			#print reflect_beam['detectorY']
			#print direct_beam['M_beampos'], direct_beam['detectorZ']
			#print omega * 180/np.pi
			
			M_twotheta += self.reflect_beam.detectorZ[:, np.newaxis, np.newaxis]
			M_twotheta += np.arange(numypixels * 1.)[np.newaxis, np.newaxis, :] * pn.Y_PIXEL_SPACING
			
			M_twotheta -= self.direct_beam.M_beampos[:, :, np.newaxis] + self.direct_beam.detectorZ
			M_twotheta /= self.reflect_beam.detectorY[:, np.newaxis, np.newaxis]
			M_twotheta = np.arctan(M_twotheta)

			if omega[0,0] < 0:
				omega = 0 - omega
				M_twotheta = 0 - M_twotheta
		elif mode == 'SB' or mode == 'DB':
			omega = self.reflect_beam.M_beampos + self.reflect_beam.detectorZ[:, np.newaxis]
			omega -= self.direct_beam.M_beampos + self.direct_beam.detectorZ
			omega /= 2 * self.reflect_beam.detectorY[:, np.newaxis, np.newaxis]
			omega = np.arctan(omega)   
			
			M_twotheta += np.arange(numypixels * 1.)[np.newaxis, np.newaxis, :] * pn.Y_PIXEL_SPACING
			M_twotheta += self.reflect_beam.detectorZ[:, np.newaxis, np.newaxis]
			M_twotheta -= self.direct_beam.M_beampos[:, :, np.newaxis] + self.direct_beam.detectorZ
			M_twotheta -= self.reflect_beam.detectorY[:, np.newaxis, np.newaxis] * np.tan(omega[:, :, np.newaxis])
			
			M_twotheta /= self.reflect_beam.detectorY[:, np.newaxis, np.newaxis]
			M_twotheta = np.arctan(M_twotheta)
			M_twotheta += omega[:, :, np.newaxis]
	   
	#    workout corrected angle of incidence and input into offspecular calcn
		M_omega = M_twotheta / 2
	#    print "angle of inc:", omega[0]
		
		#now normalise the counts in the reflected beam by the direct beam spectrum
		#this gives a reflectivity
		#and propagate the errors, leaving the fractional variance (dr/r)^2
		#this step probably produces negative reflectivities, or NaN if M_specD is 0.
		#ALSO, 
		#M_refSD has the potential to be NaN is M_topandtail or M_spec is 0.
		M_ref, M_refSD = EP.EPdiv(self.reflect_beam.M_topandtail,
									self.reflect_beam.M_topandtailSD,
									 self.direct_beam.M_spec[:, :, np.newaxis],
									  self.direct_beam.M_specSD[:, :, np.newaxis])
		
		#you may have had divide by zero's.
		M_ref = np.where(np.isinf(M_ref), 0, M_ref)
		M_refSD = np.where(np.isinf(M_refSD), 0, M_refSD)
		
		#now calculate the Q values for the detector pixels.  Each pixel has different 2theta and different wavelength, ASSUME that they have the same angle of incidence
		M_qz = np.empty_like(M_twotheta)
		M_qy = np.empty_like(M_twotheta)
		
		M_qz[:] = 2 * np.pi / self.reflect_beam.M_lambda[:, :, np.newaxis]
		M_qz *= np.sin(M_twotheta - omega[:, :, np.newaxis]) + np.sin(M_omega)
		M_qy[:] = 2 * np.pi / self.reflect_beam.M_lambda[:, :, np.newaxis]
		M_qy *= np.cos(M_twotheta - omega[:, :, np.newaxis]) - np.cos(M_omega)

	#    M_qz, M_qy = qtrans.to_qzqy(np.reshape(omega, (numspectra, numtpixels, 1)),
	#                                 M_twotheta,
	#                                  np.reshape(reflect_beam['M_lambda'], (numspectra, numtpixels, 1)))
		
		#now calculate the full uncertainty in Q for each Q pixel
		M_qzSD = np.zeros_like(M_qz)
		M_qzSD += np.reshape((self.reflect_beam.M_lambdaSD / self.reflect_beam.M_lambda)**2, (numspectra, numtpixels, 1))
		M_qzSD += (self.reflect_beam.domega[:, np.newaxis, np.newaxis] / M_omega)**2
		M_qzSD = np.sqrt(M_qzSD)
		M_qzSD *= M_qz
		
		#scale reflectivity by scale factor
		#M_ref, M_refSD = EP.EPdiv(M_ref, M_refSD, scalefactor, 0)
		
		#now calculate the 1D output
		W_q = qtrans.to_q(omega, self.reflect_beam.M_lambda)
		W_qSD = (self.reflect_beam.M_lambdaSD / self.reflect_beam.M_lambda)**2
		W_qSD += (self.reflect_beam.domega[:, np.newaxis] / omega) ** 2
		W_qSD = np.sqrt(W_qSD) * W_q
		
		lopx, hipx = self.reflect_beam.lopx, self.reflect_beam.hipx
		
		W_ref = np.sum(M_ref[:, :, lopx:hipx + 1], axis = 2)
		W_refSD = np.sum(np.power(M_refSD[:, :, lopx:hipx + 1], 2), axis = 2)
		W_refSD = np.sqrt(W_refSD)
		
		self.W_q = W_q
		self.W_qSD = W_qSD
		self.W_ref = W_ref
		self.W_refSD = W_refSD
		self.M_ref = M_ref
		self.M_refSD = M_refSD
		self.M_qz = M_qz
		self.M_qy = M_qy
		self.numspectra = numspectra
 
 def sanitize_string_input(file_list_string):
    """
    given a string like '1 2 3 4 1000 -1 sijsiojsoij' return an integer list where the numbers are greater than 0 and less than 9999999
    it strips the string.ascii_letters and any string.punctuation, and converts all the numbers to ints.
    """
    return [int(x) for x in file_list_string.translate(None, string.punctuation).translate(None, string.ascii_letters).split() if 0 < int(x) < 9999999]
 
 def reduce_stitch_files(reflect_list, direct_list, normfilenumber = None, **kwds):
	"""
	kwds passed onto processnexusfile
	"""
	scalefactor = kwds.get('scalefactor', 1.)
		   
	#now reduce all the files.
	zipped = zip(reflect_list, direct_list)

	combineddataset = reflectdataset.ReflectDataset()
	
	if kwds.get('basedir'):
		self.basedir = kwds.get('basedir')
	else:
		self.basedir = os.getcwd()
	
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
		reflectdatafilename =rdfn
		directdatafilename = ddfn
				
		for root, dirs, files in os.walk(kwds['basedir']) while not (len(reflectdatafilename) and len(directdatafilename)):
			if rdfn in files:
				reflectdatafilename = os.path.join(root, rdfn)
			if ddfn in files:
				directdatafilename = os.path.join(root, ddfn)	

		with h5py.File(reflectdatafilename, 'r') as h5ref and h5py.File(directdatafilename, 'r') as h5direct:
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
		a.write_reflectivity_XML(f)

	print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())        
	
	