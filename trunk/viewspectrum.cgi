#!/usr/local/bin/python
import os
import os.path
import sys
import cgi, cgitb
import StringIO
import h5py
import numpy as np
from pyplatypus import reduce
from pyplatypus import processplatypusnexus
from pyplatypus import platypusspectrum
import gviz_api
import zipfile
import string

cgitb.enable()

#specify where the files are to originate from
FILEPATH = './'

def main():
    form = cgi.FieldStorage()
    keys = form.keys()
    
    kwds = {}
    kwds['lolambda'] = 2.8
    kwds['hilambda'] = 18.0
    kwds['rebinpercent'] = 4.0
    
    spectra = []

    if 'spectrum' in keys:
        spectrum_list = reduce.sanitize_string_input(form.getlist('spectrum'))
        
        background = True
        for key in keys:
            if key in kwds:
                kwds[key] = float(form[key].value)
		
        red = processplatypusnexus.ProcessPlatypusNexus()
                
        try:
            for specnumber in spectrum_list:
                specname = ''
                sn = 'PLP{0:07d}.nx.hdf'.format(int(abs(specnumber)))
                for root, dirs, files in os.walk(FILEPATH):
                    if sn in files:
                        specname = os.path.join(root, sn)
                        break
                
                if not len(specname):
                    continue
                
                with h5py.File(specname, 'r') as h5data:
                    spectrum = red.process(h5data, **kwds)
                    spectra.append(spectrum)
        except:
            print "Content-type: text/plain\n"
            print 'a'
            return
			
    if not len(spectra):
        print "Content-type: text/plain\n"
        print ''
        return
	
    if 'JSON' in keys:
        print "Content-type: text/plain\n"
        print spectra_to_json(spectra)
        return
    else:
        zipped_spectra = spectra_to_zip(spectra)
        length = len(zipped_spectra.getvalue())
        
        datacontent = '\r\n'.join(["Content-type: %s;",
                            "Content-Disposition: attachment; filename=\"%s\"",
                            "Content-Title: %s",
                            "Content-Length: %i",
                            "",])

        print datacontent % ('application/octet-stream', 'data.zip', 'data.zip', length)
        sys.stdout.write(zipped_spectra.getvalue()) # *not* print, doh !
        sys.stdout.flush()
		
def spectra_to_json(spectra):
	'''
		
		spectra is a list of the processed spectra
		Returns a json representation of the spectra, suitable for google visualisation
		
	'''
	
	numspectra = len(spectra)
	data = []
	description = [('lamda', 'number')]

	for index, spectrum in enumerate(spectra):
		description.append((str(spectrum.datafilenumber), 'number'))
         
		wavelength = spectrum.M_lambda[0]
		intensity = spectrum.M_spec[0]
		numpoints = np.size(wavelength, axis = 0)
		
		for index2 in xrange(numpoints):
			record = [None] * (numspectra + 1)
			record[0] = wavelength[index2]
			record[index + 1] = intensity[index2]
			data.append(record)
 
	data_table = gviz_api.DataTable(description)
	data_table.LoadData(data)

	json = data_table.ToJSon()
	
	return json
	
def spectra_to_zip(spectra):
    '''
    
    spectra is a list of the processed spectra
    returns a zip file with the zipped spectra files.
    
    '''
    
    theZipFile = StringIO.StringIO()
    
    with zipfile.ZipFile(theZipFile, 'w') as Tzipfile:
        for spectrum in spectra:
            spectrum_file = StringIO.StringIO()
            spectrum.write_spectrum_XML(spectrum_file)
            filename = 'PLP{:07d}.spectrum'.format(spectrum.datafilenumber)

            Tzipfile.writestr(filename, spectrum_file.getvalue())
    
    return theZipFile
		

if __name__ == '__main__':
    main()
