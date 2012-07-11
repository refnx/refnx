#!/usr/local/bin/python
import os
import sys
import cgi, cgitb
import StringIO
import h5py
import numpy as np
from pyplatypus import reduce
from pyplatypus import processplatypusnexus
from pyplatypus import platypusspectrum
import gviz_api

cgitb.enable()

#specify where the files are to originate from
FILEPATH = './'

def main():
    form = cgi.FieldStorage()
    keys = form.keys()
    print "Content-type: text/plain\n"
    
    kwds = {}
    kwds['lolambda'] = 2.8
    kwds['hilambda'] = 18.0
    kwds['rebinpercent'] = 4.0
    
    if 'spectrum' in keys:
        spectrum_list = reduce.sanitize_string_input(form['spectrum'].value)
        
        background = True
        for key in keys:
            if key in kwds:
                kwds[key] = float(form[key].value)
		
        red = processplatypusnexus.ProcessPlatypusNexus()
        specname = ''
                
        description = [('lamda', 'number')]
        spectra = []
        for specnumber in spectrum_list:
            sn = 'PLP{0:07d}.nx.hdf'.format(int(abs(specnumber)))
            for root, dirs, files in os.walk(FILEPATH):
                if sn in files:
                    specname = os.path.join(root, sn)
                    break
			
            if not len(specname):
                continue
                                
            with h5py.File(specname, 'r') as h5data:
                spectrum = red.process(h5data, **kwds)
        
            description.append((sn, 'number'))
            spectra.append(spectrum)
         
        if not len(spectra):
            print ''
            return

        numspectra = len(spectra)
        data = []

        for index, val in enumerate(spectra):            
            wavelength = val.M_lambda[0]
            intensity = val.M_spec[0]
            numpoints = np.size(wavelength, axis = 0)
            
            for index2 in xrange(numpoints):
                record = [None] * (numspectra + 1)
                record[0] = wavelength[index2]
                record[index + 1] = intensity[index2]
                data.append(record)
 
#        description = [('lamda', 'number'), ('I', 'number')]
#        ydata = np.log10(spectrum.M_spec[0])     
#        data = zip(spectrum.M_lambda[0], ydata)
        
 			
        data_table = gviz_api.DataTable(description)
        data_table.LoadData(data)
        
        json = data_table.ToJSon()
        print json
    
if __name__ == '__main__':
    main()
