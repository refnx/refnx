#!/usr/local/bin/python
import os
import os.path
import sys
import cgi, cgitb
import StringIO
import h5py
import numpy as np
from pyplatypus.reduce import reduce
from pyplatypus.reduce import reflectdataset
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
    kwds['background'] = True
	
    for key in keys:
        if key in kwds:
            kwds[key] = float(form[key].value)

    if 'reflect_spectrum' in keys:
        reflect_spectrum_list = reduce.sanitize_string_input(form.getlist('reflect_spectrum'))	
	
    if 'direct_spectrum' in keys:
        direct_spectrum_list = reduce.sanitize_string_input(form.getlist('direct_spectrum'))	

    spectra_pairs = zip(reflect_spectrum_list, direct_spectrum_list)
	
    if 'normfilenumber' in keys and len(form.getlist('normfilenumber')):
        normfilenumber = reduce.sanitize_string_input(form.getlist('normfilenumber'))[0]
    else:
        normfilenumber = None

	kwds['basedir'] = FILEPATH
	
	try:
		combineddataset = reduce.reduce_stitch(reflect_spectrum_list, direct_spectrum_list, normfilenumber = normfilenumber)
		combineddataset.rebin(rebinpercent = kwds['rebinpercent'])
	except:
		print "Content-type: text/plain\n"
		print 'a'
		return
           
    if 'JSON' in keys:
        print "Content-type: text/plain\n"
        print reflectivity_to_json(reflect_spectrum_list[0], combineddataset)
        return		
    else:
        theSpectrum = StringIO.StringIO()
        combineddataset.write_reflectivity_XML(theSpectrum)
        length = len(theSpectrum.getvalue())
        
        filename = 'c_PLP{:07d}.xml'.format(reflect_spectrum_list[0], 0)

        datacontent = '\r\n'.join(["Content-type: %s;",
                            "Content-Disposition: attachment; filename=\"%s\"",
                            "Content-Title: %s",
                            "Content-Length: %i",
                            "",])

        print datacontent % ('text/xml', filename, filename, length)

        sys.stdout.write(theSpectrum.getvalue()) # *not* print, doh !
        sys.stdout.flush()
		
def reflectivity_to_json(filenumber, combineddataset):
	'''
		
		Returns a json representation of the combined dataset, suitable for google visualisation
		
	'''
	
	description = [('Q', 'number'), (str(filenumber), 'number')]	
	data_table = gviz_api.DataTable(description)

	qq, RR, dR, dq = combineddataset.get_data()
	data = zip(qq, RR)
 
	data_table.LoadData(data)
	
	return data_table.ToJSon()
	
		

if __name__ == '__main__':
    main()
