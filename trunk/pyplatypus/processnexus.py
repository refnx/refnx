from __future__ import division
import numpy as np
import h5py as h5
import  ErrorProp as EP
import utility as ut
import Qtransforms as qtrans
from scipy.optimize import curve_fit
from scipy.stats import t
import zlib
import rebin
import string
from time import gmtime, strftime
import os


#opening of the choppers, in radians
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
Y_PIXEL_SPACING = 1.177    #in mm
CHOPFREQ = 23                #Hz
ROUGH_BEAM_POSITION = 150        #Rough direct beam position
ROUGH_BEAM_WIDTH = 10
CHOPPAIRING = 3
    
def processnexusfile(datafilenumber, **kwds):
    """ basedir = None,
        lolambda = 2.8, hilambda = 18., background = True, normfilename = None,
        eventstreaming = None, isdirect = False, peak_pos = None,
        typeofintegration = 0, expected_width = 10, omega = 0, two_theta = 0, rebinpercent = 4,
        bmon1_normalise = True):
    """  
  
    
    """
     processes a Nexus file
     type of integration = 0 means sum all spectra
    type of integration != 0 means output spectra individually
    """
    
    datafilename = 'PLP{0:07d}.nx.hdf'.format(int(abs(datafilenumber)))
    
    if 'basedir' in kwds:
        for root, dirs, files in os.walk(kwds['basedir']):
            if datafilename in files:
                datafilename = os.path.join(root, datafilename)
                break
               
    lolambda = kwds.get('lolambda', 2.8)
    hilambda = kwds.get('hilambda', 18.)    
    background = kwds.get('background', True)
    normfilename = kwds.get('normfilename', None)
    eventstreaming = kwds.get('eventstreaming', None)
    isdirect = kwds.get('isdirect', False)
    peak_pos = kwds.get('peak_pos', None)
    typeofintegration = kwds.get('typeofintegration', 0)
    expected_width = kwds.get('expected_width', 10.)
    omega = kwds.get('omega', 0.)
    two_theta = kwds.get('two_theta', 0.) 
    rebinpercent = kwds.get('rebinpercent', 4.) 
    bmon1_normalise = kwds.get('bmon1_normalise', True) 

    try:
        h5data = h5.File(datafilename, 'r')
    except IOerror:
        return None
        
    scanpoint = 0
    
    #beam monitor counts for normalising data
    bmon1_counts = np.zeros(dtype = 'float64', shape = h5data['entry1/monitor/bm1_counts'].shape)
    bmon1_counts = h5data['entry1/monitor/bm1_counts'][:]
    
    #set up the RAW TOF bins (n, t + 1)
    TOF = np.zeros(dtype = 'float64', shape = h5data['entry1/data/time_of_flight'].shape)
    TOF = h5data['entry1/data/time_of_flight'][:]

    #event streaming.
    if eventstreaming:
        scanpoint = eventstreaming['scanpoint']
        detector = get_detector_from_streamed_data(eventstreaming['compressed_data_stream'],
                                    tbins = h5data['entry1/data/x_bin'] * 1000,
                                    xbins = h5data['entry1/data/x_bin'],
                                    ybins = h5data['entry1/data/y_bin'],
                                    frame_agg = eventstreaming['agg_number'])
           
        numspectra = len(detector)
        bmon1 = bmon1_counts[scanpoint]
        bmon1_counts = np.resize(bmon1_counts, numspectra)
        bmon1_counts[:] = bmon1 / numspectra
    else:
        #detector(n, t, y, x)    
        detector = np.zeros(dtype='int32', shape = h5data['entry1/data/hmm'].shape)
        detector = h5data['entry1/data/hmm'][:]
        
        #you average over the individual measurements in the file
        if typeofintegration == 0:
            numspectra = 1
            detector = np.sum(detector, axis = 0)
            detector = np.resize(detector, (1, np.size(detector, 0), np.size(detector, 1), np.size(detector, 2)))
            bmon1_counts = np.array(np.sum(bmon1_counts), dtype = 'float64')
        else:
            numspectra = len(detector)

    #pre-average over x, leaving (n, t, y)
    detector = np.sum(detector, axis = 3, dtype = 'float64')

    #detector shape should now be (n, t, y)
    #create the SD of the array
    detectorSD = np.sqrt(detector)
    
    #detector normalisation with a water file
    if normfilename:
        xbins = h5data['entry1/data/x_bin']
        #shape (y,)
        M_detectornorm, M_detectornormSD = createdetectornorm(normfilename, xbins[0], xbins[1])
        #detector has shape (n,t,y), shape of M_waternorm should broadcast to (1,1,y)
        detector, detectorSD = EP.EPdiv(detector, detectorSD, M_detectornorm, M_detectornormSD)
        detectorSD = np.where(np.isfinite(detectorSD) == 0, 0, detectorSD)
        
    #get the specular ridge on the averaged detector image
    if peak_pos:
        beam_centre, beam_SD = peak_pos
    else:
        beam_centre, beam_SD = findspecularridge(detector)
        #print "BEAM_CENTRE", datafilenumber, beam_centre
    
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
        
        if eventstreaming:
            M_specTOFHIST[:] = TOF - toffset
            chod[:] = chod[0]
            detpositions[:] = detpositions[0]
            break
        else:
            scanpoint += 1
    
    scanpoint = originalscanpoint
        
    #convert TOF to lambda
    #M_specTOFHIST (n, t) and chod is (n,)
    M_lambdaHIST = qtrans.tof_to_lambda(M_specTOFHIST, chod.reshape(numspectra, 1))
    M_lambda = 0.5 * (M_lambdaHIST[:,1:] + M_lambdaHIST[:,:-1])
    TOF -= toffset
    
    assert not np.isnan(detectorSD).any()
    assert not np.less(detectorSD, 0).any()
    
    #TODO gravity correction if direct beam
    if isdirect:
        detector, detectorSD, M_gravcorrcoefs = correct_for_gravity(detector, detectorSD, M_lambda, 0, 2.8, 18)
        beam_centre, beam_SD = findspecularridge(detector)
        
    #rebinning in lambda for all detector
    #rebinning is the default option, but sometimes you don't want to.
    #detector shape input is (n, t, y)
    #we want to rebin t.
    if 0 < rebinpercent < 10.:
        frac = 1. + (rebinpercent/100.)
        lowl = (2 * lolambda) / ( 1. + frac)
        hil =  frac * (2 * hilambda) / ( 1. + frac)
               
        numsteps = np.log10(hil / lowl ) // np.log10(frac)
        rebinning = np.logspace(np.log10(lowl), np.log10(hil), num = numsteps + 2)

        rebinneddata = np.zeros((numspectra, np.size(rebinning, 0) - 1, np.size(detector, 2)), dtype = 'float64')
        rebinneddataSD = np.zeros((numspectra, np.size(rebinning, 0) - 1, np.size(detector, 2)), dtype = 'float64')
        
        for index in xrange(np.size(detector, 0)):
        #rebin that plane.
            plane, planeSD = rebin.rebin2D(M_lambdaHIST[index], np.arange(np.size(detector, 2) + 1.),
                detector[index], detectorSD[index], rebinning, np.arange(np.size(detector, 2) + 1.))
            assert not np.isnan(planeSD).any()
                
            rebinneddata[index, ] = plane
            rebinneddataSD[index, ] = planeSD
        
        detector = rebinneddata
        detectorSD = rebinneddataSD
    
        M_lambdaHIST = np.resize(rebinning, (numspectra, np.size(rebinning, 0)))
   
    M_specTOFHIST = qtrans.lambda_to_tof(M_lambdaHIST, chod.reshape(numspectra, 1))
    M_lambda = 0.5 * (M_lambdaHIST[:,1:] + M_lambdaHIST[:,:-1])
    M_spectof = qtrans.lambda_to_tof(M_lambda, chod.reshape(numspectra, 1))
    
    #Now work out where the beam hits the detector    #this is used to work out the correct angle of incidence.	#it will be contained in a wave called M_beampos	#M_beampos varies as a fn of wavelength due to gravity
		
    #TODO work out beam centres for all pixels
    #this has to be done agian because gravity correction is done above.
    if isdirect:
       #the spectral ridge for the direct beam has a gravity correction involved with it.       #the correction coefficients for the beamposition are contaned in M_gravcorrcoefs
        M_beampos = np.zeros_like(M_lambda)
		        # the following correction assumes that the directbeam neutrons are falling from a point position 
        # W_gravcorrcoefs[0] before the detector. At the sample stage (W_gravcorrcoefs[0] - detectorpos[0])
        # they have a certain vertical velocity, assuming that the neutrons had
        # an initial vertical velocity of 0. Although the motion past the sample stage will be parabolic,
        # assume that the neutrons travel in a straight line after that (i.e. the tangent of the parabolic
        # motion at the sample stage). This should give an idea of the direction of the true incident beam,
        # as experienced by the sample.        #Factor of 2 is out the front to give an estimation of the increase in 2theta of the reflected beam.        M_beampos[:] = M_gravcorrcoefs[:,1].reshape(numspectra, 1)
        M_beampos[:] -= 2. * (1000. / Y_PIXEL_SPACING * 9.81 * ((M_gravcorrcoefs[:, 0].reshape(numspectra, 1) - detpositions.reshape(numspectra, 1))/1000.) * (detpositions.reshape(numspectra, 1)/1000.) * M_lambda**2/((qtrans.kPlanck_over_MN * 1.e10)**2))        M_beampos *=  Y_PIXEL_SPACING
    else:
        M_beampos = np.zeros_like(M_lambda)
        M_beampos[:] = beam_centre * Y_PIXEL_SPACING

    #background subtraction
    extent_mult = 2.4
    if background:
        detector, detectorSD = background_subtract(detector, detectorSD, beam_centre, beam_SD, extent_mult, 2)
    
    #top and tail the specular beam with the known beam centres.
    #all this does is produce a specular intensity with shape (n, t), i.e. integrate over specular beam
    lopx = np.floor(beam_centre - beam_SD * extent_mult)
    hipx = np.ceil(beam_centre + beam_SD  * extent_mult)

    M_spec = np.sum(detector[:, :, lopx:hipx + 1], axis = 2)
    M_specSD = np.sum(np.power(detectorSD[:, :, lopx:hipx + 1], 2), axis = 2)
    M_specSD = np.sqrt(M_specSD)
    
    #
    #normalise by beam monitor 1.
    #
    if bmon1_normalise:
        bmon1_countsSD = np.sqrt(bmon1_counts)
        #have to make to the same shape as M_spec
        bmon1_counts =  bmon1_counts.reshape(numspectra, 1)
        bmon1_countsSD = bmon1_countsSD.reshape(numspectra, 1)
        M_spec, M_specSD = EP.EPdiv(M_spec, M_specSD, bmon1_counts, bmon1_countsSD)
        #have to make to the same shape as detector
        #bmon1_counts =  np.reshape(bmon1_counts, (numspectra, 1, 1))
        bmon1_countsSD =  bmon1_countsSD.reshape(numspectra, 1, 1)
        detector, detectorSD = EP.EPdiv(detector, detectorSD, bmon1_counts, bmon1_countsSD)
        
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
    		
    #create a massive dictionary with the list of stuff that you've produced.
    output = {'M_topandtail': detector,
              'M_topandtailSD': detectorSD,
              'M_spec': M_spec,
              'M_specSD': M_specSD,
              'M_beampos': M_beampos,
              'M_lambda': M_lambda,
              'M_lambdaSD': M_lambdaSD,
              'M_lambdaHIST': M_lambdaHIST,
              'M_spectof': M_spectof,
              'mode' : mode,
              'detectorZ' : detectorZ,
              'detectorY' : detectorY,
              'domega' : domega,
              'lopx' : lopx,
              'hipx' : hipx,
              'title' : h5data['entry1/experiment/title'][0],
              'sample' : h5data['entry1/sample/name'][0],
              'user' : h5data['entry1/user/name'][0],
              'runnumber' : datafilenumber
              }
    #remember to close the hdf file
    h5data.close()
    return output
    
def spectrum_XML(runnumber, M_spec, M_specSD, M_lambda, M_lambdaSD, title = ''):
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
    d['title'] = title
    d['time'] = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
    #sort the data
    sorted = np.argsort(M_lambda[0])
    r = M_spec[:,sorted]
    l = M_lambda[:, sorted]
    dl = M_lambdaSD[:, sorted]
    dr = M_specSD[:, sorted]
    d['numpoints'] = np.size(r, axis = 1)
    for index in xrange(len(M_spec)):
        filename = 'PLP{:07d}_{:d}.spectrum'.format(runnumber, index)
                    
        d['r'] = string.translate(repr(r[index].tolist()), None, ',[]')
        d['dr'] = string.translate(repr(dr[index].tolist()), None, ',[]')
        d['l'] = string.translate(repr(l[index].tolist()), None, ',[]')
        d['dl'] = string.translate(repr(dl[index].tolist()), None, ',[]')
        f = open(filename, 'w')
        thefile = s.safe_substitute(d)
        f.write(thefile)
        f.truncate()
        f.close()

def get_detector_from_streamed_data(compressed_data_stream, 
                                    tbins = None, xbins = None, ybins = None,
                                    frame_agg = 1200):
    """
    reads a zippedunpackedbin string and converts into a detector image
    compressed_data_stream is a zlib compressed data stream.  The uncompressed stream has a 128
    bit header.  Then the structure is:
        [int16,int16, uint32, uint32, uint8, uint8, int16] * N
    corresponding to:
        [x,    y,     t,      f,      v,     w,     spare]
        (t has units of us)
    tbins, xbins and ybins are needed to control how the events are constructed into the detector image
    frame_agg controls how many frames are added together.
    """
    
#    dec = zlib.decompressobj()
#    dec.decompress(compressed_data_stream, 16)
#    
#    xx = np.empty(0, dtype = 'int16')
#    yy = np.empty(0, dtype = 'int16')
#    tt = np.empty(0, dtype = 'uint32')
#    ff = np.empty(0, dtype = 'uint32')
#    
#    while dec.unconsumed_tail:
#        an_event = dec.decompress(dec.unconsumed_tail, 16)
#        x, y, t, f, other = struct.unpack('hhIII',an_event)
#        np.append
                
    data = np.fromstring(zlib.decompress(compressed_data_stream, dtype='uint32'))
    tt = np.copy(data[5::4])
    ff = np.copy(data[6::4])
    
    numevents = len(tt) 
    xx = np.asarray(data[4::4] & 0x0000FFFF, dtype = 'int16')
    yy = np.asarray(data[4::4] >> 16, dtype = 'int16')
    
    del(data)
    
    total_frames = ff[-1] 
    numspectra = np.ceil(total_frames / frame_agg)
    frame_bins = np.arange(numspectra + 1, dtype = 'uint32') * frame_agg
    detector, edges = np.histogramdd((ff, tt, yy, xx), bins = (frame_bins, tbins, ybins, xbins))
    return detector
    
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
    mode = h5data['entry1/instrument/parameters/mode'][scanpoint]
	
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
        chod -= chopper2_distance[scanpoint]
    elif master == 3:
        chod -= chopper3_distance[scanpoint]
	
		
    if slave == 2:
        chod -= chopper2_distance[scanpoint]
    elif slave == 3:
        chod -= chopper3_distance[scanpoint]
    elif slave == 4:
        chod -= chopper4_distance[scanpoint]
	
			
    #T0 is midway between master and slave, but master may not necessarily be disk 1.
    #However, all instrument lengths are measured from disk1
    chod /= 2
    
    if mode == "FOC" or mode == "POL" or mode == "MT" or mode == "POLANAL":
        chod += sample_distance[scanpoint]
        chod += detectorpos[scanpoint] / np.cos(np.pi * two_theta / 180)
    
    elif mode == "SB":   		
        #assumes guide1_distance is in the MIDDLE OF THE MIRROR
        chod += guide1_distance[scanpoint]
        chod += (sample_distance[scanpoint] - guide1_distance[scanpoint]) / np.cos(np.pi * omega / 180)
        if two_theta > omega:
            chod += detectorpos[scanpoint]/np.cos( np.pi* (two_theta - omega) / 180)
        else:
            chod += detectorpos[scanpoint] /np.cos( np.pi * (omega - two_theta) /180)
    
    elif mode == "DB":
        #guide2_distance in in the middle of the 2nd compound mirror
        # guide2_distance - longitudinal length from midpoint1->midpoint2 + direct length from midpoint1->midpoint2
        chod += guide2_distance[scanpoint] + 600. * np.cos (1.2 * np.pi/180) * (1 - np.cos(2.4 * np.pi/180)) 
    
        #add on distance from midpoint2 to sample
        chod +=  (sample_distance[scanpoint] - guide2_distance[scanpoint]) / np.cos(4.8 * np.pi/180)
        
        #add on sample -> detector			
        if two_theta > omega:			
            chod += detectorpos[scanpoint] / np.cos( np.pi* (two_theta - 4.8) / 180)
        else:
            chod += detectorpos[scanpoint] / np.cos( np.pi * (4.8 - two_theta) /180)
    
    return chod
    
def findspecularridge(nty_wave, tolerance = 0.01):
    """
    find the specular ridge in a detector(n, t, y) plot.
    """
    
    searchincrement = 50
    #sum over all n planes, left with ty
    det_ty = np.sum(nty_wave, axis = 0)
    
    #find a good place to start the peak search from
    totaly = np.sum(det_ty, axis = 0)
    centroid, gausscentre = ut.peakfinder(totaly)
    
    lastcentre = centroid[0]
    lastSD = centroid[1]
    
    numincrements = len(det_ty) // searchincrement
        
    for ii in xrange(numincrements):
        totaly = np.sum(det_ty[-1: -1 - searchincrement * (ii + 1): -1], axis = 0)
        #find the centroid and gauss peak in the last sections of the TOF plot
        centroid, gausspeak = ut.peakfinder(totaly)
        if abs((gausspeak[0] - lastcentre) / lastcentre) < tolerance and abs((gausspeak[1] - lastSD) / lastSD) < tolerance:
            lastcentre = gausspeak[0]
            lastSD = gausspeak[1]
            break
        
        lastcentre = gausspeak[0]
        lastSD = gausspeak[1]
    
    return lastcentre, lastSD
    
def correct_for_gravity(data, dataSD, lamda, trajectory, lolambda, hilambda):
	#this function provides a gravity corrected yt plot, given the data, its associated errors, the wavelength corresponding to each of the time bins, and the trajectory of the neutrons.  Low lambda and high Lambda are wavelength cutoffs to igore.		#output:	#corrected data, dataSD	#M_gravCorrCoefs.  THis is a theoretical prediction where the spectral ridge is for each timebin.  This will be used to calculate the actual angle of incidence in the reduction process.
	
	#data has shape (n, t, y)
	#M_lambda has shape (n, t)
    numlambda = np.size(lamda, axis = 1)
	
    x_init = np.arange((np.size(data, axis = 2) + 1) * 1.) - 0.5
    
    f = lambda x, td, tru_centre: deflection(x, td, 0) / Y_PIXEL_SPACING + tru_centre
    
    M_gravcorrcoefs = np.zeros((len(data), 2), dtype = 'float64')
    
    correcteddata = np.empty_like(data)
    correcteddataSD = np.empty_like(dataSD) 

    for spec in xrange(len(data)):
        #centres(t,)
        centroids = np.apply_along_axis(ut.centroid, 1, data[spec])
        lopx = np.trunc(np.interp(lolambda, lamda[spec], np.arange(numlambda)))
        hipx = np.ceil(np.interp(hilambda, lamda[spec], np.arange(numlambda)))
        
    	M_gravcorrcoefs[spec], pcov = curve_fit(f, lamda[spec,lopx:hipx], centroids[:, 0][lopx:hipx], np.array([3000., np.mean(centroids)]))
    	totaldeflection = deflection(lamda[spec], M_gravcorrcoefs[spec][0], 0) / Y_PIXEL_SPACING

    	for wavelength in xrange(np.size(data, axis = 1)):
    	    x_rebin = x_init + totaldeflection[wavelength]
            correcteddata[spec,wavelength], correcteddataSD[spec,wavelength] = rebin.rebin(x_init, data[spec,wavelength], dataSD[spec, wavelength], x_rebin)
    
    return correcteddata, correcteddataSD, M_gravcorrcoefs

def deflection(lamda, travel_distance, trajectory):	#returns the deflection in mm of a ballistic neutron	#lambda in Angstrom, travel_distance (length of correction, e.g. sample - detector) in mm, trajectory in degrees above the horizontal	#The deflection correction  is the distance from where you expect the neutron to hit the detector (detector_distance*tan(trajectory)) to where is actually hits the detector, i.e. the vertical deflection of the neutron due to gravity.	trajRad = trajectory*np.pi/180	pp = travel_distance/1000. * np.tan(trajRad)
	
	pp -= 9.81* (travel_distance/1000.)**2 * (lamda/1.e10)**2 / (2*np.cos(trajRad)*np.cos(trajRad)*(qtrans.kPlanck_over_MN)**2)	pp *= 1000	return pp
    
def createdetectornorm(normfilename, xmin, xmax):
    """
    produces a detector normalisation for Platypus
    you give it a water run and it average n, t and x to provide 
    a relative efficiency for each y wire.
    """
    h5norm = h5.File(normfilename, 'r')
    
    #average over n and TOF and x
    #n
    norm = np.sum(np.array(h5data['entry1/data/hmm']), axis = 0, dtype = 'float64')
    #t
    norm = np.sum(detector, axis = 0, dtype = 'float64')
    # by this point you have norm[y][x]
    norm = norm[:, xmin:xmax + 1]
    norm = np.sum(norm, axis = 1, dtype = 'float64')    
        
    normSD = np.empty_like(norm)
    normSD = np.sqrt(norm)
    
    norm /= mean
    normSD /= mean
    
    h5norm.close()
    return norm, normSD

def background_subtract(detector, detectorSD, beam_centre, beam_SD, extent_mult = 2.2, pixel_offset = 2):
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
    
def background_subtract_line(detector, detectorSD, beam_centre, beam_SD, extent_mult = 2.2, pixel_offset = 2):
    
    lopx = np.floor(beam_centre - beam_SD * extent_mult)
    hipx = np.ceil(beam_centre + beam_SD  * extent_mult)
    
    y0 = round( lopx - (extent_mult * beam_SD) - pixel_offset )
    y1 = round(lopx - pixel_offset)
    y2 = round(hipx + pixel_offset)
    y3 = round(hipx + (extent_mult * beam_SD) + pixel_offset)
    
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
    if np.isfinite(pcov).any():
        bkgdSD = np.zeros_like(bkgd)
    else:
        bkgdSD = np.asarray([CI(x, pcov) for x in np.arange(len(detector))], dtype = 'float64')
        
    bkgdSD = np.sqrt(bkgdSD)
    #get the t value for a two sided student t test at the 68.3 confidence level
    
    bkgdSD *= t.isf(0.1585, len(xvals) - 2)
    
    return EP.EPsub(detector, detectorSD, bkgd, bkgdSD)
