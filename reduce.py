from __future__ import division
import numpy as np
import processnexus as pro_nex
import ErrorProp as EP
import Qtransforms as qtrans
import string
import reflect_dataset as rd
import os
import tempfile
import shutil
import zipfile
import hashlib
import quickplot
import sys

def reduce_stitch_files(reflect_list, direct_list, scalefactor = 1., collect = False, basedir = None):
    cwd = os.getcwd()
    filename = ''
    tempdir = ''
    try:
        #first process all the reflected beam spectra
        reflected_runs = [pro_nex.processnexusfile(file_number, basedir = basedir) for file_number in reflect_list]
        #now all the direct beam spectra
        direct_runs = [pro_nex.processnexusfile(file_number, basedir = basedir) for file_number in direct_list]
            
        if collect:
            tempdir = tempfile.mkdtemp()
            masterdir = os.path.join(tempdir, 'master')
            spectrumdir = os.path.join(tempdir, 'master', 'spectrum')
            reflectdir = os.path.join(tempdir, 'master', 'reflect')
            stitchdir = os.path.join(reflectdir, 'stitch')
            os.mkdir(masterdir)
            os.mkdir(spectrumdir)
            os.mkdir(reflectdir)
            os.mkdir(stitchdir)
            os.chdir(spectrumdir)
        
        #write out all the spectra
        for index, val in enumerate(reflected_runs):
            pro_nex.spectrum_XML(val['runnumber'],
                                  val['M_spec'],
                                    val['M_specSD'],
                                        val['M_lambda'],
                                            val['M_lambdaSD'],
                                                val['title'])
            
        for index, val in enumerate(direct_runs):
            pro_nex.spectrum_XML(val['runnumber'],
                                  val['M_spec'],
                                   val['M_specSD'],
                                    val['M_lambda'],
                                     val['M_lambdaSD'],
                                      val['title'])
        
       
        #now reduce all the files.
        zipped = zip(reflected_runs, direct_runs)
        ref_runs, direct_runs = zip(*zipped)
        reduced_data = map(reduce_single_file, ref_runs, direct_runs)
        
        if collect:
            os.chdir(reflectdir)
        #write them all to file and create the stitched file
        for index, val in enumerate(reduced_data):
            #write the specular datasets
            dataset = rd.Reflect_Dataset(val.get1Ddata(), Reduced_Output = val)
            dataset.scale(scalefactor = scalefactor)
            dataset.write_xml()
            dataset.write_dat()
            #write the offspecular datasets
            dataset_2D = rd.Reflect_Dataset_2D(val.M_qz, val.M_qy, val.M_ref, val.M_refSD, val.reflect_beam_spectrum['runnumber'])
            dataset_2D.write_xml()
            #stitch the individual runs together
            if index:
                stitchedData.add_dataset(val)
            else:
                stitchedData = rd.Reflect_Dataset(val.get1Ddata(), Reduced_Output = val)
                stitchedData.scale(scalefactor = scalefactor)
            
        #now get the stitched file
        if len(reduced_data) > 1:
            if collect:
                os.chdir(stitchdir)
            
            stitchedData.rebin()
            stitchedData.write_xml()
            stitchedData.write_dat()
            #print out a picture
            W_q, W_ref, W_refSD, W_qSD = stitchedData.getData()
            name = 'c_PLP{:07d}.png'.format(stitchedData._rnumber[0])
            quickplot.the_ref_plot(name, W_q[0], W_ref[0], W_refSD[0])
        
        if collect:
            os.chdir(tempdir)
            shutil.make_archive('data','zip', masterdir)

            m = hashlib.md5()       
            with open(os.path.join(tempdir,'data.zip'), 'r') as f:
                for stuff in f.read():
                    m.update(stuff)

            filename = m.hexdigest() + '.zip'
            shutil.copyfile(os.path.join(tempdir,'data.zip'), os.path.join(cwd, filename))
            
    except IOError as io:
        sys.stderr.write('ERROR - reduce_stitch_files - one of the hdf5 files was missing')
    except Exception as e:
        print repr(e)
  
    os.chdir(cwd)
    shutil.rmtree(tempdir, True)

    return filename
    

def reduce_single_file(reflect_beam, direct_beam, scalefactor = 1):
    #num files to output.
    M_topandtail = reflect_beam['M_topandtail']
    numspectra = len(reflect_beam['M_spec'])
    numtpixels = np.size(M_topandtail, 1)
    numypixels = np.size(M_topandtail, 2)
    
    #calculate omega and two_theta depending on the mode.
    mode = reflect_beam['mode'][0]
    M_twotheta = np.zeros(M_topandtail.shape, dtype = 'float64')
    
    if mode == 'FOC' or mode == 'POL' or mode == 'POLANAL' or mode == 'MT':
        # (n,t)            (n,t)                            (n, )           
        omega = reflect_beam['M_beampos'] + np.reshape(reflect_beam['detectorZ'], (numspectra, 1))
        #            (1, t)                     (1, )
        omega -= direct_beam['M_beampos'] + direct_beam['detectorZ']
        #                (n, ) 
        omega /= np.reshape(reflect_beam['detectorY'], (numspectra, 1))

        omega = np.arctan(omega) / 2
        #print reflect_beam['M_beampos'], reflect_beam['detectorZ']
        #print reflect_beam['detectorY']
        #print direct_beam['M_beampos'], direct_beam['detectorZ']
        #print omega * 180/np.pi
        
        #(n, t, y)                    (n, )                                                      
        M_twotheta += np.reshape(reflect_beam['detectorZ'], (numspectra, 1, 1))
        #             (1, 1, y)
        M_twotheta += np.reshape(np.arange(numypixels * 1.), (1, 1, numypixels)) * pro_nex.Y_PIXEL_SPACING
        #                (1, t)                                                        (1, )
        M_twotheta -= np.reshape(direct_beam['M_beampos'], (1, numtpixels, 1)) + direct_beam['detectorZ']
        
        #(n, t, y)                        (n,)
        M_twotheta /= np.reshape(reflect_beam['detectorY'], (numspectra, 1, 1))
        M_twotheta = np.arctan(M_twotheta)

        if omega[0,0] < 0:
            omega = 0 - omega
            M_twotheta = 0 - M_twotheta
    elif mode == 'SB' or mode == 'DB':
        omega = reflect_beam['M_beampos'] + np.reshape(reflect_beam['detectorZ'], (numspectra, 1))
        omega -= direct_beam['M_beampos'] + direct_beam['detectorZ']
        omega /= 2 * np.reshape(reflect_beam['detectorY'], (numspectra, 1))
        omega = np.arctan(omega)   
        
        M_twotheta += np.reshape(np.arange(numypixels * 1.), (1, 1, numypixels)) * pro_nex.Y_PIXEL_SPACING
        M_twotheta += np.reshape(reflect_beam['detectorZ'], (numspectra, 1, 1))
        M_twotheta -= np.reshape(direct_beam['M_beampos'], (1, numtpixels, 1)) + direct_beam['detectorZ']
        M_twotheta -= np.reshape(reflect_beam['detectorY'], (numspectra, 1, 1)) * np.tan(np.reshape(omega, (numspectra, numtpixels, 1)))
        
        M_twotheta /= np.reshape(reflect_beam['detectorY'], (numspectra, 1, 1))
        M_twotheta = np.arctan(M_twotheta)
        M_twotheta += np.reshape(omega, (numspectra, numtpixels, 1))
   
#    TODO workout corrected angle of incidence and input into offspecular calcn
    M_omega = M_twotheta / 2
    
    #now normalise the counts in the reflected beam by the direct beam spectrum
    #this gives a reflectivity
    #and propagate the errors, leaving the fractional variance (dr/r)^2
    #this step probably produces negative reflectivities, or NaN if M_specD is 0.
    #ALSO, 
    #M_refSD has the potential to be NaN is M_topandtail or M_spec is 0.
    
    #        (n, t, y)        (1, t)    
    M_ref, M_refSD = EP.EPdiv(M_topandtail,
                                reflect_beam['M_topandtailSD'],
                                 np.reshape(direct_beam['M_spec'], (1, numtpixels, 1)),
                                  np.reshape(direct_beam['M_specSD'], (1, numtpixels, 1)))

    #you may have had divide by zero's.
    M_ref = np.where(np.isinf(M_ref), 0, M_ref)
    M_refSD = np.where(np.isinf(M_refSD), 0, M_refSD)
    
    #now calculate the Q values for the detector pixels.  Each pixel has different 2theta and different wavelength, ASSUME that they have the same angle of incidence
    #(n, t, y)
    #TODO workout gravity corrected angle of incidence and input into offspecular calcn
    M_qz, M_qy = qtrans.to_qzqy(np.reshape(omega, (numspectra, numtpixels, 1)),
                                 M_twotheta,
                                  np.reshape(reflect_beam['M_lambda'], (numspectra, numtpixels, 1)))
    
    #now calculate the full uncertainty in Q for each Q pixel
    #(n, t, y)       (n, t)                        (n, t)
    M_qzSD = np.zeros_like(M_qz)
    M_qzSD += np.reshape((reflect_beam['M_lambdaSD'] / reflect_beam['M_lambda'])**2, (numspectra, numtpixels, 1))
    #                               (n,)                                (n, t, y)
    M_qzSD += (np.reshape(reflect_beam['domega'], (numspectra, 1, 1)) / M_omega)**2
    M_qzSD = np.sqrt(M_qzSD)
    M_qzSD *= M_qz
    
    #scale reflectivity by scale factor
    M_ref, M_refSD = EP.EPdiv(M_ref, M_refSD, scalefactor, 0)
    
    #now calculate the 1D output
    W_q = qtrans.to_q(omega, reflect_beam['M_lambda'])
    W_qSD = (reflect_beam['M_lambdaSD'] / reflect_beam['M_lambda'])**2
    W_qSD += (np.reshape(reflect_beam['domega'], (numspectra, 1)) / omega) ** 2
    W_qSD = np.sqrt(W_qSD) * W_q
    
    lopx, hipx = reflect_beam['lopx'], reflect_beam['hipx']
    
    W_ref = np.sum(M_ref[:, :, lopx:hipx + 1], axis = 2)
    W_refSD = np.sum(np.power(M_refSD[:, :, lopx:hipx + 1], 2), axis = 2)
    W_refSD = np.sqrt(W_refSD)
    
    output = { 'W_q' : W_q,
               'W_qSD' : W_qSD,
               'W_ref' : W_ref,
               'W_refSD' : W_refSD,
               'M_ref' : M_ref,
               'M_refSD' : M_refSD,
               'M_qz' : M_qz,
               'M_qzSD' : M_qzSD,
               'M_qy' : M_qy,
               'reflect_beam_spectrum' : reflect_beam,
               'direct_beam_spectrum' : direct_beam}
    obj = rd.Reduced_Output(output)    
    return obj 
    
